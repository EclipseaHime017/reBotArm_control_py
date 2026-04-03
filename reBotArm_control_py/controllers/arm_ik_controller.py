"""ArmIK — 纯 IK posvel组合控制器。（速度位置控制）

使用:
    arm = RobotArm()
    arm_ik = ArmIK(arm)
    arm_ik.start()
    arm_ik.move_to_ik(x=0.3, y=0.1, z=0.4)         # 默认零姿态
    arm_ik.move_to_ik(x=0.4, y=0.2, z=0.3, roll=0.0, pitch=1.57, yaw=0.0)
    arm_ik.end()
"""

from __future__ import annotations

import numpy as np

from ..kinematics import (
    compute_ik,
)
from ..actuator import RobotArm


class ArmIK:

    def __init__(self, arm: RobotArm) -> None:
        self.arm = arm
        self._n = arm.num_joints
        self._q = np.zeros(self._n)
        self._q_target = np.zeros(self._n)
        self._pv_vlim = np.array([j.vlim for j in arm._joints], dtype=np.float64)
        self._running = False

    def start(self) -> None:
        """连接、切换模式、使能、启动控制循环。"""
        self.arm.connect()
        self.arm.mode_pos_vel()
        self.arm.enable()
        self.arm.start_control_loop(self._loop_cb)
        self._running = True

    def move_to_ik(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> bool:
        """IK 求解并驱动机械臂移动到目标位姿。"""
        if not self._running:
            return False

        result = compute_ik(
            q_init=self._q,
            target_pos=np.array([x, y, z]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        if not result.success:
            print(f"[ArmIK] IK 未收敛  err={result.error:.3e}")
            return False

        self._q_target = result.q.copy()
        return True

    def end(self) -> None:
        """断开连接。"""
        if self._running:
            self.arm.disconnect()
            self._running = False

    # ── 内部 ──────────────────────────────────────────────────────────────────

    def _loop_cb(self, _: RobotArm, dt: float) -> None:
        try:
            pos = self.arm.get_positions()
            if pos is not None and len(pos) == self._n:
                self._q[:] = np.asarray(pos, dtype=np.float64)
        except Exception:
            pass
        self.arm.pos_vel(self._q_target, vlim=self._pv_vlim)

    def __enter__(self) -> "ArmIK":
        return self

    def __exit__(self, *args) -> None:
        self.end()