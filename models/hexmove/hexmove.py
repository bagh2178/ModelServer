import os
import time
import math
import rclpy
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from scipy.spatial.transform import Rotation as R
from piper_sdk import *
from .utils import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
# from .realsense import capture_rgbd_image, capture_intrinsic
from .realsense import D435i, T265
from .orbbec import OrbbecCamera
from .odom_subscriber import get_odom_pose, get_odom_xy_and_yaw, get_camera_xy_and_yaw
# from .odom_subscriber import OdomSubscriber
from .point_cloud_generator import generate_point_cloud


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


class Hexmove():
    def __init__(self) -> None:
        self.save_image_dir = '/home/tl/yh/RGBD/images/{}'
        self.camera_list = {
            'D435i_top': {
                'serial_number': '337322070914',
                'R_robot_to_camera': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                'T_robot_to_camera': np.array([0.35, 0, 1.3]),
            },
            'D435i_down': {
                'serial_number': '327122078142',
                'R_robot_to_camera': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                'T_robot_to_camera': np.array([0.43, 0, 0.88]),
            },
            'T265': {
                'serial_number': '119622110447',
                'R_robot_to_camera': np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                'T_robot_to_camera': np.array([0.35, 0, 0]),
            },
            'FemtoBolt_down': {
                'serial_number': 'CL8M841006W',
                # 'R_robot_to_camera': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                'R_robot_to_camera': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ R.from_euler('xyz', (-np.pi / 6, 0, 0)).as_matrix(),
                'T_robot_to_camera': np.array([0.43, 0, 0.88]),
            },
            '336L_down': {
                'serial_number': 'CP828410001R',
                'R_robot_to_camera': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                'T_robot_to_camera': np.array([0.35, 0, 1.0]),
            },
        }
        self.tracking_method = 'odom'
        self.tracking_method = 'T265'

    def __call__(self, commond):
        # rclpy.spin_once(self.odom_subscriber, timeout_sec=0.1)
        action = commond[0]
        if action == 'get_rgbd_intrinsic':
            camera_id = commond[1]
            self.init_camera(camera_id)
            color_intrinsics, depth_intrinsics = self.camera_list[camera_id]['camera'].capture_intrinsic()
            return color_intrinsics, depth_intrinsics
        elif action == 'get_rgbd_image':
            camera_id = commond[1]
            serial_number = self.camera_list[camera_id]['serial_number']
            self.init_camera(camera_id)
            if 'pose' in commond:
                pose_list, timestamp_list = self.record_camera_pose(camera_id=camera_id, record_time=0.5)
                rgb_image, depth_image, timestamp = self.camera_list[camera_id]['camera'].capture_rgbd_image()
                closest_index = self.timestamp_match(timestamp_list, timestamp)
                pose = pose_list[closest_index]
                camera_param = self.camera_list[camera_id]['camera'].capture_camera_param()
                self.save_image(serial_number, rgb_image, depth_image, camera_param, pose, timestamp)
                return rgb_image, depth_image, pose, timestamp
            else:
                rgb_image, depth_image, timestamp = self.camera_list[camera_id]['camera'].capture_rgbd_image()
                self.save_image(serial_number, rgb_image, depth_image, timestamp)
                return rgb_image, depth_image, timestamp
        elif action == 'get_tracking_pose':
            position, orientation, timestamp = self.get_tracking_pose()
            return position, orientation, timestamp
        elif action == 'get_robot_pose':
            position, orientation, timestamp = self.get_robot_pose()
            return position, orientation, timestamp
        elif action == 'get_robot_xy_and_yaw':
            x, y, yaw = self.get_robot_xy_and_yaw()
            return x, y, yaw
        elif action == 'get_camera_pose':
            camera_id = commond[1]
            camera_position, camera_orientation, timestamp = self.get_camera_pose(camera_id)
            return camera_position, camera_orientation, timestamp
        elif action == 'get_camera_xy_and_yaw':
            camera_id = commond[1]
            x, y, yaw = self.get_camera_xy_and_yaw(camera_id)
            return x, y, yaw
        elif action == 'get_pc':
            camera_id = commond[1]
            self.init_camera(camera_id)
            serial_number = self.camera_list[camera_id]['serial_number']
            # if camera_id in ['D435i_top', 'D435i_down']:
            #     self.camera_list[camera_id]['camera'] = D435i(serial_number)
            if camera_id in ['FemtoBolt_down', '336L_down']:
                pose_list, timestamp_list = self.record_camera_pose(camera_id=camera_id, record_time=0.5)
                points, timestamp = self.camera_list[camera_id]['camera'].capture_color_point_cloud()
                closest_index = self.timestamp_match(timestamp_list, timestamp)
                pose = pose_list[closest_index]
                position, orientation = pose
                points[:, :3] = (R.from_quat(quat_wxyz_to_xyzw(orientation)).as_matrix() @ points[:, :3].T).T + position
            # rgb_image, depth_image = self.camera_list[camera_id]['camera'].capture_rgbd_image()
            # x, y, yaw = self.get_robot_xy_and_yaw()

            # 相机内参
            # color_intrinsics, depth_intrinsics = self.camera_list[camera_id]['camera'].capture_intrinsic()
            # fx, fy, cx, cy = 388., 388., 320., 240.  # 请根据实际情况调整
            # fx, fy, cx, cy = color_intrinsics
            # depth_scale = 0.001  # 将毫米转换为米

            # 生成点云
            # point_cloud_world, points_world, colors = generate_point_cloud(depth_image, rgb_image, fx, fy, cx, cy, depth_scale, x, y, yaw, self.camera_list[camera_id]['R_robot_to_camera'], self.camera_list[camera_id]['T_robot_to_camera'])

            # 保存点云为TXT文件（带颜色）
            output_file_path_world = os.path.join(self.save_image_dir.format(serial_number), 'pc')
            if not os.path.exists(output_file_path_world):
                os.makedirs(output_file_path_world)
            output_file_path_world = os.path.join(output_file_path_world, f'{timestamp}.txt')
            # np.savetxt(output_file_path_world, np.hstack((points_world, colors)), fmt='%.6f', delimiter=' ', header='X Y Z R G B')
            np.savetxt(output_file_path_world, points, fmt='%.6f', delimiter=' ', header='X Y Z R G B')

            print(f"Point cloud with colors in world coordinate system saved to {output_file_path_world}")

            return 'get_pc_top done' 
        elif action == 'move_forward':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1')
            return 'move_forward done'
        elif action == 'move_backward':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1')
            return 'move_backward done'
        elif action == 'move_left':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 1.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1')
            return 'move_left done'
        elif action == 'move_right':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: -1.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1')
            return 'move_right done'
        elif action == 'turn_left':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}" -1')
            return 'turn_left done'
        elif action == 'turn_right':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -1.0}}" -1')
            return 'turn_right done'
        elif action == 'move_forward_1':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 100 -t 75')
            return 'move_forward_1 done'
        elif action == 'move_backward_1':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 100 -t 75')
            return 'move_backward_1 done'
        elif action == 'move_left_1':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 1.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 100 -t 75')
            return 'move_left_1 done'
        elif action == 'move_right_1':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: -1.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 100 -t 75')
            return 'move_right_1 done'
        elif action == 'turn_left_90':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}" -r 100 -t 120')
            return 'turn_left_90 done'
        elif action == 'turn_right_90':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -1.0}}" -r 100 -t 120')
            return 'turn_right_90 done'
        elif action == 'turn_back':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}" -r 100 -t 300')
            return 'turn_back done'
        elif action == 'turn_360':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}" -r 100 -t 650')
            return 'turn_360 done'
        elif action == 'stop':
            return 'success'
        elif action == 'get_arm_end_pose':
            arm_id = commond[1]
            piper = C_PiperInterface(arm_id)
            piper.ConnectPort()
            pose = piper.GetArmEndPoseMsgs()
            return pose
        elif action == 'arm_move':
            arm_id = commond[1]
            piper = C_PiperInterface(arm_id)
            piper.ConnectPort()
            piper.EnableArm(7)
            enable_fun(piper=piper)
            piper.GripperCtrl(0, 1000, 0x01, 0)
            factor = 1000
            position = commond[2]
            X = round(position[0] * factor)
            Y = round(position[1] * factor)
            Z = round(position[2] * factor)
            RX = round(position[3] * factor)
            RY = round(position[4] * factor)
            RZ = round(position[5] * factor)
            joint_6 = round(position[6]*factor)
            piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        else:
            return commond
        
    def init_camera(self, camera_id):
        if 'camera' not in self.camera_list[camera_id]:
            serial_number = self.camera_list[camera_id]['serial_number']
            if camera_id in ['D435i_top', 'D435i_down']:
                self.camera_list[camera_id]['camera'] = D435i(serial_number)
            elif camera_id in ['T265']:
                self.camera_list[camera_id]['camera'] = T265(serial_number)
            elif camera_id in ['FemtoBolt_down', '336L_down']:
                self.camera_list[camera_id]['camera'] = OrbbecCamera(serial_number)

    def save_image(self, serial_number, rgb_image, depth_image, camera_param=None, pose=None, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        rgb_image_path = os.path.join(self.save_image_dir.format(serial_number), 'rgb', f'{timestamp}.png')
        depth_image_path = os.path.join(self.save_image_dir.format(serial_number), 'depth', f'{timestamp}.png')
        Image.fromarray(rgb_image).save(rgb_image_path)
        Image.fromarray(depth_image, mode='I;16').save(depth_image_path)
        if camera_param is not None:
            camera_param_path = os.path.join(self.save_image_dir.format(serial_number), 'intrinsic', f'{timestamp}.txt')
            with open(camera_param_path, 'w') as file:
                file.write(camera_param.__str__())
        if pose is not None:
            pose_path = os.path.join(self.save_image_dir.format(serial_number), 'extrinsic', f'{timestamp}.txt')
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R.from_quat(quat_wxyz_to_xyzw(pose[1])).as_matrix()
            extrinsic[:3, 3] = pose[0]
            np.savetxt(pose_path, extrinsic)

    def get_tracking_pose(self):
        self.init_camera(self.tracking_method)
        position, orientation, timestamp = self.camera_list[self.tracking_method]['camera'].get_pose()
        return position, orientation, timestamp
    
    def get_robot_pose(self):
        if self.tracking_method == 'odom':
            position, orientation, timestamp = get_odom_pose()
        elif self.tracking_method == 'T265':
            tracking_position, tracking_orientation, timestamp = self.get_tracking_pose()
            R_robot_to_camera = self.camera_list[self.tracking_method]['R_robot_to_camera']
            T_robot_to_camera = self.camera_list[self.tracking_method]['T_robot_to_camera']
            R_robot_to_camera = R.from_matrix(R_robot_to_camera).inv().as_matrix()
            T_robot_to_camera = R_robot_to_camera @ (-T_robot_to_camera)
            orientation = R.from_matrix(R_robot_to_camera) * R.from_quat(quat_wxyz_to_xyzw(tracking_orientation)) * R.from_matrix(R_robot_to_camera)
            orientation = quat_xyzw_to_wxyz(orientation.as_quat())
            # orientation = orientation.as_euler('xyz')
            position = T_robot_to_camera + R_robot_to_camera @ (tracking_position + R.from_quat(quat_wxyz_to_xyzw(tracking_orientation)).as_matrix() @ T_robot_to_camera)
        self.robot_position = position
        self.robot_orientation = orientation
        return position, orientation, timestamp
    
    def get_robot_xy_and_yaw(self):
        position, orientation, timestamp = self.get_robot_pose()
        position_x, position_y = position[0], position[1]
        roll, pitch, yaw = R.from_quat(quat_wxyz_to_xyzw(orientation)).as_euler('xyz')
        return position_x, position_y, yaw
    
    def get_camera_pose(self, camera_id):
        position, orientation, timestamp = self.get_robot_pose()
        camera_position = position + R.from_quat(quat_wxyz_to_xyzw(orientation)).as_matrix() @ self.camera_list[camera_id]['T_robot_to_camera']
        camera_orientation = R.from_quat(quat_wxyz_to_xyzw(orientation)) * R.from_matrix(self.camera_list[camera_id]['R_robot_to_camera'])
        camera_orientation = quat_xyzw_to_wxyz(camera_orientation.as_quat())
        return camera_position, camera_orientation, timestamp
    
    def get_camera_xy_and_yaw(self, camera_id):
        position, orientation, timestamp = self.get_camera_pose(camera_id)
        position_x, position_y = position[0], position[1]
        roll, pitch, yaw = R.from_quat(quat_wxyz_to_xyzw(self.robot_orientation)).as_euler('xyz')
        return position_x, position_y, yaw
    
    def record_camera_pose(self, camera_id, record_time=1.0):
        pose_list = []
        timestamp_list = []
        self.init_camera(self.tracking_method)
        tracking_fps = self.camera_list[self.tracking_method]['camera'].get_fps()
        num_record_frame = int(tracking_fps * record_time)
        for i in range(num_record_frame):
            camera_position, camera_orientation, timestamp = self.get_camera_pose(camera_id)
            pose_list.append((camera_position, camera_orientation))
            timestamp_list.append(timestamp)
        return pose_list, timestamp_list
    
    def timestamp_match(self, timestamp_list, timestamp):
        min_diff = float('inf')
        closest_index = -1

        for i, ts in enumerate(timestamp_list):
            diff = abs(ts - timestamp)

            if diff < min_diff:
                min_diff = diff
                closest_index = i

        return closest_index
