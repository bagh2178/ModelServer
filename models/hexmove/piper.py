import time
import numpy as np
from piper_sdk import *


def enable_fun(piper:C_PiperInterface):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)


class PiperArm():
    def __init__(self, can=None):
        self.can = can
        self.piper = C_PiperInterface(self.can)
        self.piper.ConnectPort()
        self.end_pose_factor = 1000
        self.joint_factor = 57324.840764
        self.gripper_pose_factor = 1000 * 60
        self.pose_origin = [55, 0, 206, 0, 85, 0, 0]
        self.pose_prepare = [
            [200, 0, 300, 0, 90, 0, 0],
            [250, 0, 200, 0, 110, 0, 0],
            [300, 0, 100, 0, 120, 0, 0],
            [300, 0, 0, 0, 120, 0, 0],
        ]
        self.is_enable = False

    def reset(self):
        self.arm_end_pose_ctrl(self.pose_origin)

    def prepare(self):
        for pose in self.pose_prepare:
            self.arm_end_pose_ctrl(pose)

    def enable(self):
        if not self.is_enable:
            self.piper.EnableArm(7)
            enable_fun(piper=self.piper)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
            self.is_enable = True

    def disable(self):
        if self.is_enable:
            self.reset()
            time.sleep(3)
            self.piper.DisableArm(7)
            self.is_enable = False

    def get_arm_pose(self):
        arm_end_pose = self.piper.GetArmEndPoseMsgs()
        arm_joint = self.piper.GetArmJointMsgs()
        arm_gripper_pose = self.piper.GetArmGripperMsgs()
        timestamp = arm_end_pose.time_stamp
        arm_end_pose = np.array([arm_end_pose.end_pose.X_axis, arm_end_pose.end_pose.Y_axis, arm_end_pose.end_pose.Z_axis, arm_end_pose.end_pose.RX_axis, arm_end_pose.end_pose.RY_axis, arm_end_pose.end_pose.RZ_axis])
        arm_end_pose = arm_end_pose / self.end_pose_factor
        arm_joint = np.array([arm_joint.joint_state.joint_1, arm_joint.joint_state.joint_2, arm_joint.joint_state.joint_3, arm_joint.joint_state.joint_4, arm_joint.joint_state.joint_5, arm_joint.joint_state.joint_6])
        arm_joint = arm_joint / self.joint_factor
        arm_gripper_pose = arm_gripper_pose.gripper_state.grippers_angle / self.gripper_pose_factor
        arm_pose = np.append(arm_end_pose, arm_joint)
        arm_pose = np.append(arm_pose, arm_gripper_pose)
        return arm_end_pose, arm_joint, arm_gripper_pose, timestamp
    
    def arm_gripper_ctrl(self, arm_gripper_pose):
        self.enable()
        joint_6 = round(arm_gripper_pose * self.gripper_pose_factor)
        self.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)
        self.piper.GripperCtrl(abs(joint_6), 5000, 0x01, 0)
        return 'done'
    
    def arm_end_pose_ctrl(self, arm_end_pose):
        self.enable()
        for i in range(100):
            X = round(arm_end_pose[0] * self.end_pose_factor)
            Y = round(arm_end_pose[1] * self.end_pose_factor)
            Z = round(arm_end_pose[2] * self.end_pose_factor)
            RX = round(arm_end_pose[3] * self.end_pose_factor)
            RY = round(arm_end_pose[4] * self.end_pose_factor)
            RZ = round(arm_end_pose[5] * self.end_pose_factor)
            joint_6 = round(arm_end_pose[6] * self.gripper_pose_factor)
            self.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            self.piper.GripperCtrl(abs(joint_6), 5000, 0x01, 0)
            time.sleep(0.01)
        return 'done'

    def arm_joint_ctrl(self, arm_joint):
        self.enable()
        for i in range(100):
            joint_0 = round(arm_joint[0] * self.joint_factor)
            joint_1 = round(arm_joint[1] * self.joint_factor)
            joint_2 = round(arm_joint[2] * self.joint_factor)
            joint_3 = round(arm_joint[3] * self.joint_factor)
            joint_4 = round(arm_joint[4] * self.joint_factor)
            joint_5 = round(arm_joint[5] * self.joint_factor)
            joint_6 = round(arm_joint[6] * self.gripper_pose_factor)
            self.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            self.piper.GripperCtrl(abs(joint_6), 5000, 0x01, 0)
            time.sleep(0.01)
        return 'done'