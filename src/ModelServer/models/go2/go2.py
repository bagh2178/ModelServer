from .robot_interface import GO2Interface
from .gui_controller import GUIController
from .gopro.gopro10 import init_gopro10


class Go2:
    def __init__(self):
        # receive the robot perception information in one process
        self.robot = GO2Interface('eth0')
        self.robot.subscribe_process()

        self.gopro10 = init_gopro10(width=224, height=224)

        print('go2 initialized')

    def remote_control_loop(self):
        self.controller = GUIController(self.robot)
        self.controller.rollout()
    
    def get_gopro_frame(self, num_frames=1):
        return self.gopro10.read_frame(num_frames)
    
    def get_pc(self):
        pc = self.robot.get_pointcloud() 
        return pc
    
    def get_imu(self):
        robot_state = self.robot.get_sportstate()
        return robot_state.imu_state
    
    def get_pose(self):
        return
    
    def get_joint(self):
        low_state = self.robot.get_lowstate()
        joint_pose = [low_state.motor_state[i].q for i in range(20)]
        return joint_pose
    
    def set_joint(self, joint_pos):
        self.robot.joint_control(joint_pos)
    
    def set_coordinate(self, x, y):
        self.robot.get_to_coordination_goal(x, y)
    
    def get_arm_joint(self):
        low_state = self.robot.get_lowstate()
        arm_joint_pose = [low_state.motor_state[i].q for i in range(10)]
        return arm_joint_pose
    
    def set_arm_joint(self, arm_joint_pos):
        self.robot.arm_joint_control(arm_joint_pos)
    
