"""ArmTraj — 轨迹规划组合控制器。

使用:
    arm = RobotArm()
    arm_traj = ArmTraj(arm)
    arm_traj.start()
    arm_traj.move_to_traj(x=0.3, y=0.0, z=0.3, roll=0, pitch=0.4, yaw=0, duration=2.0)
    arm_traj.end()
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..kinematics import (
    compute_fk,
    pos_rot_to_se3,
    get_end_effector_frame_id,
    load_robot_model,
)
from ..kinematics.inverse_kinematics import (
    solve_ik,
    IKParams as TrajIKParams,
)
from ..trajectory import (
    TrajProfile,
    TrajPlanParams,
    IKParams as ClikIKParams,
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
)
from ..actuator import RobotArm


class ArmTraj:

    def __init__(
        self,
        arm: RobotArm,
        dt: float = 0.02,
        profile: TrajProfile = TrajProfile.MIN_JERK,
    ) -> None:
        self.arm = arm
        self._n = arm.num_joints
        self._dt = dt
        self._model = load_robot_model()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._data = self._model.createData()

        self._pv_vlim = np.array([j.vlim for j in arm._joints], dtype=np.float64)

        self._traj_params = TrajPlanParams(dt=dt, profile=profile)
        self._ik_solver_params = TrajIKParams(
            max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6,
        )
        self._ik_params = ClikIKParams(
            max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8,
        )

        # 轨迹状态（由规划线程写入，由 loop_cb 读取）
        self._q_target = np.zeros(self._n)
        self._traj: list[np.ndarray] = []
        self._traj_idx = 0

        self._running = False
        self._send_thread: threading.Thread | None = None
        self._stop_send = threading.Event()

    def start(self) -> None:
        """连接、切换模式、使能、启动控制循环。"""
        self.arm.connect()
        self.arm.mode_pos_vel()
        self.arm.enable()
        self.arm.start_control_loop(self._loop_cb)
        self._running = True

    def move_to_traj(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        """SE(3) 测地线规划 + CLIK 跟踪，驱动机械臂沿平滑轨迹移动。"""
        if not self._running:
            return False

        # ── Step 1: 读取实机当前关节位置 ─────────────────────────────────
        pos = self.arm.get_positions(request=True)
        if pos is None or len(pos) != self._n:
            print("[ArmTraj] 无法读取关节位置")
            return False
        q_start = np.asarray(pos, dtype=np.float64)
        print(f"[ArmTraj] q_start (deg): {np.degrees(q_start).round(1).tolist()}")

        # ── Step 2: 目标 SE(3) 位姿 ──────────────────────────────────────
        T_target = pos_rot_to_se3(
            np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw,
        )

        # ── Step 3: IK（从真实起点出发） ─────────────────────────────────
        ik_result = solve_ik(
            self._model, self._data, self._end_frame_id,
            T_target, q_start, self._ik_solver_params,
        )
        if not ik_result.success:
            print(f"[ArmTraj] IK 失败，error={ik_result.error:.4f}")
            return False

        q_end = ik_result.q
        print(f"[ArmTraj] q_end   (deg): {np.degrees(q_end).round(1).tolist()}")

        # ── Step 4: 真实端点位姿 ─────────────────────────────────────────
        T_start = compute_fk(self._model, q_start)[2]
        T_end = compute_fk(self._model, q_end)[2]

        # ── Step 5: 自动时长估算 ────────────────────────────────────────
        if duration <= 0:
            dist = float(np.linalg.norm(T_target.translation() - T_start.translation()))
            duration = max(1.0, dist / 0.1)

        # ── Step 6: SE(3) 测地线采样 ─────────────────────────────────────
        cart_traj = plan_cartesian_geodesic_trajectory(
            T_start, T_end, duration, self._traj_params,
        )

        # ── Step 7: CLIK 跟踪 ────────────────────────────────────────────
        joint_traj = track_trajectory(
            self._model, self._end_frame_id,
            cart_traj.trajectory, q_start, self._ik_params,
            null_gain=0.1,
        )
        if not joint_traj:
            print("[ArmTraj] 轨迹为空")
            return False

        # ── Step 8: 启动延迟发送线程 ──────────────────────────────────────
        pts = [pt.q.copy() for pt in joint_traj]
        print(f"[ArmTraj] 轨迹点数={len(pts)}  duration={duration:.2f}s")

        self._stop_send.set()
        if self._send_thread is not None:
            self._send_thread.join()

        self._traj = pts
        self._traj_idx = 0
        self._stop_send.clear()
        self._send_thread = threading.Thread(
            target=self._send_loop, args=(duration,), daemon=True,
        )
        self._send_thread.start()

        return True

    def end(self) -> None:
        """断开连接。"""
        if self._running:
            self._stop_send.set()
            self.arm.disconnect()
            self._running = False

    # ── 控制循环：高频发送当前目标角度 ──────────────────────────────────

    def _loop_cb(self, _: RobotArm, dt: float) -> None:
        self.arm.pos_vel(self._q_target, vlim=self._pv_vlim)

    # ── 轨迹发送线程：按 dt=20ms 节奏写入目标角度 ───────────────────────

    def _send_loop(self, duration: float) -> None:
        n = len(self._traj)
        interval = duration / n if n > 0 else self._dt
        for i in range(n):
            if self._stop_send.is_set():
                return
            self._q_target[:] = self._traj[i]
            time.sleep(interval)
