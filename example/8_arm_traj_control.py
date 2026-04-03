#!/usr/bin/env python3
"""轨迹规划交互控制示例。

用法:
    python example/8_arm_traj_control.py

输入:
    x y z [roll pitch yaw] [duration]   目标末端位置（米 / 弧度 / 秒）
    q / quit / exit                     退出
    state                               当前状态
    pos                                 当前末端位置
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.controllers import ArmTraj


def main() -> None:
    arm = RobotArm()
    arm_traj = ArmTraj(arm)

    arm_traj.start()
    print("--- 已启动轨迹控制器 ---\n")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break

        if line.lower() == "state":
            q, _, _ = arm.get_state()
            print(f"  当前关节 (rad): {[f'{v:+.3f}' for v in q]}")
            print(f"  moving: {arm_traj._moving}  "
                  f"traj_pts: {len(arm_traj._joint_traj)}  "
                  f"idx: {arm_traj._traj_idx}")
            continue

        if line.lower() == "pos":
            q, _, _ = arm.get_state()
            from reBotArm_control_py.kinematics import joint_to_pose
            pos, rpy = joint_to_pose(q)
            print(f"  pos=[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}] m  "
                  f"rpy=[{rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}] rad")
            continue

        try:
            vals = [float(v) for v in line.split()]
        except ValueError:
            print("  格式: x y z [roll pitch yaw] [duration]")
            continue

        x, y, z = vals[0], vals[1], vals[2]
        roll = vals[3] if len(vals) >= 6 else 0.0
        pitch = vals[4] if len(vals) >= 6 else 0.0
        yaw = vals[5] if len(vals) >= 6 else 0.0
        duration = vals[6] if len(vals) >= 7 else 2.0

        ok = arm_traj.move_to_traj(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            duration=duration,
        )
        print(f"  -> ({x:+.3f}, {y:+.3f}, {z:+.3f})  "
              f"T={duration:.1f}s  {'ok' if ok else 'fail'}")

    arm_traj.end()
    print("\n完成。")


if __name__ == "__main__":
    main()
