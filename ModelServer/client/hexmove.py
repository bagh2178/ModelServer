from ..tcp import TCPClient


class Hexmove_Client():
    '''
    向 Hexmove 机器人发送命令.
    支持的命令包括：
        'get_rgbd_image', <camera_id>, <format>='png', <save>='save', <pose>='pose', <without_depth>='without_depth'
        'get_pointcloud', <camera_id>
        'get_camera_intrinsic', <camera_id>
        'get_camera_extrinsic', <camera_id>
        'get_camera_xy_and_yaw', <camera_id>

        'robot_pose_reset'
        'get_robot_pose'

        'robot_move', <goal>
        'robot_move_openloop', <goal>

        'get_arm_pose', <arm_id>

        'arm_enable', <arm_id>
        'arm_disable', <arm_id>
        'arm_reset', <arm_id>
        'arm_prepare', <arm_id>
        'arm_open_gripper', <arm_id>
        'arm_close_gripper', <arm_id>
        'arm_move_camera', <camera_id>, <arm_id>, <position>, <orientation>=None
        'arm_move_robot', <arm_id>, <position>, <orientation>=None
        'arm_move_local', <arm_id>, <position>, <orientation>=None
        'arm_end_pose_ctrl', <arm_id>, <arm_end_pose>
        'arm_joint_ctrl', <arm_id>, <arm_joint>

    Examples:
        >>> agent = Hexmove_Client()
        >>> agent('get_robot_pose')
        Robot Pose:
        [[1. 0. 0. x]
         [0. 1. 0. y]
         [0. 0. 1. z]
         [0. 0. 0. 1.]]
        Timestamp: 1679856000.123456
        >>> agent('get_rgbd_image', 'FemtoBolt_down')
        <rgb_image>, <depth_image>, <timestamp>
        >>> agent('get_rgbd_image', 'FemtoBolt_down', 'save', 'pose')
        <rgb_image>, <depth_image>, <pose>, <timestamp>
        >>> agent('arm_move_camera', 'FemtoBolt_down', 'arm_right', [0.139, 0.067, 0.330])
        'done'
    '''
    def __init__(self, server_ip='166.111.73.73', server_port=7002):
        self.server_ip = server_ip
        self.server_port = server_port
        self.tcpclient = TCPClient(self.server_ip, self.server_port)

    def __call__(self, *x):
        response = self.send_and_receive(*x)
        return response
    
    def send(self, *x):
        self.tcpclient.send_data(*x)

    def send_and_receive(self, *x):
        response = self.tcpclient.send_and_receive_data(*x)
        return response
