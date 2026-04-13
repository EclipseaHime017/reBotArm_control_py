"""actuator 模块 - 直接基于 motorbridge SDK 的机械臂控制。

使用示例::

    from reBotArm_control_py.actuator import RobotArm

    arm = RobotArm("config/robot.yaml")
    arm.connect()
    arm.disable()
    arm.set_zero()     
    arm.enable()
    arm.mode_mit()
    arm.mit(positions=np.array([...]), kp=np.array([...]), kd=np.array([...]))

    arm.mode_pos_vel()
    arm.pos_vel(positions=np.array([...]))

    arm.mode_vel()
    arm.set_vel(velocities=np.array([...]))

    arm.stop_control_loop()
    arm.disconnect()
"""

from .arm import RobotArm, JointCfg, load_cfg

__all__ = [
    "RobotArm",
    "JointCfg",
    "load_cfg",
]
