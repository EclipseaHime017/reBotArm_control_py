#!/usr/bin/env python3
"""ArmIK 交互控制示例。

用法:
    python example/7_arm_ik_control.py

输入:
    x y z [roll pitch yaw]   目标末端位置（米 / 弧度）
    q / quit / exit          退出
    state                    当前状态
    pos                      当前末端位置
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.controllers import ArmIK


def main() -> None:
    arm = RobotArm()
    arm_ik = ArmIK(arm)

    arm_ik.start()
    print("--- 已启动 IK 控制器 ---\n")

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
            print(f"  目标关节 (rad): {[f'{v:+.3f}' for v in arm_ik._q_target]}")
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
            print("  格式: x y z [roll pitch yaw]")
            continue

        x, y, z = vals[0], vals[1], vals[2]
        roll = vals[3] if len(vals) >= 6 else 0.0
        pitch = vals[4] if len(vals) >= 6 else 0.0
        yaw = vals[5] if len(vals) >= 6 else 0.0

        ok = arm_ik.move_to_ik(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        print(f"  -> ({x:+.3f}, {y:+.3f}, {z:+.3f})  "
              f"rpy=[{roll:+.2f} {pitch:+.2f} {yaw:+.2f}]  "
              f"{'ok' if ok else 'fail'}")

    arm_ik.end()
    print("\n完成。")


if __name__ == "__main__":
    main()