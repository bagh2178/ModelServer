import os
import io
import time
import math
import rclpy
import pickle
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from scipy.spatial.transform import Rotation as R
from piper_sdk import *
from .utils import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, timestamp_match, pose_to_matrix, matrix_to_pose
from .realsense import D435i, T265
from .orbbec import OrbbecCamera
from .piper import PiperArm
from .odom_subscriber import get_odom_pose, get_odom_xy_and_yaw, get_camera_xy_and_yaw
from .point_cloud_generator import generate_point_cloud


class Hexmove():
    def __init__(self) -> None:
        self.save_image_dir = '/home/tl/yh/ModelServer/models/hexmove/data/images/{}'
        self.save_image_dir = '/home/tl/yh/data/{}/episode_{:0>6}/rgb'
        self.save_arm_pose_dir = '/home/tl/yh/data/{}/episode_{:0>6}/pose'
        self.supported_commonds = [
            'robot_pose_reset',
            'get_rgbd_image',
            'get_rgbd_image_rdt',
            'get_pointcloud',
            'get_camera_intrinsic',
            'get_camera_extrinsic',
            'get_camera_xy_and_yaw',
            'get_robot_pose',
            'robot_move',
            'robot_move_openloop',
            'get_arm_pose',
            'get_arm_pose_rdt',
            'arm_reset',
            'arm_prepare',
            'arm_open_gripper',
            'arm_close_gripper',
            'arm_enable',
            'arm_disable',
            'arm_move_camera',
            'arm_move_robot',
            'arm_move_local',
            'arm_end_pose_ctrl',
            'arm_joint_ctrl',
        ]
        self.device_list = {
            'D435i_top': {
                'serial_number': '337322070914',
                'T_robot_to_camera': np.array([[0, 0, 1, 0.35],
                                               [-1, 0, 0, 0],
                                               [0, -1, 0, 1.3],
                                               [0, 0, 0, 1]]),
            },
            'D435i_down': {
                'serial_number': '327122078142',
                'T_robot_to_camera': np.array([[0, 0, 1, 0.43],
                                               [-1, 0, 0, 0],
                                               [0, -1, 0, 0.88],
                                               [0, 0, 0, 1]]),
            },
            'T265': {
                'serial_number': '119622110447',
                'T_robot_to_camera': np.array([[0, 0, -1, 0.35],
                                               [-1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]]),
            },
            'FemtoBolt_up': {
                'serial_number': 'CL8M841005A',
                'T_robot_to_camera': np.array([[0, 0, 1, 0.33],
                                               [-1, 0, 0, 0],
                                               [0, -1, 0, 1.3],
                                               [0, 0, 0, 1]]),
                'rotation': R.from_euler('xyz', (-np.pi / 6, 0, 0)).as_matrix()
            },
            'FemtoBolt_down': {
                'serial_number': 'CL8M841006W',
                'T_robot_to_camera': np.array([[0, 0, 1, 0.43],
                                               [-1, 0, 0, 0],
                                               [0, -1, 0, 0.88],
                                               [0, 0, 0, 1]]),
                'rotation': R.from_euler('xyz', (-np.pi / 6, 0, 0)).as_matrix()
            },
            '336L_down': {
                'serial_number': 'CP828410001R',
                'T_robot_to_camera': np.array([[0, 0, 1, 0.35],
                                               [-1, 0, 0, 0],
                                               [0, -1, 0, 1.0],
                                               [0, 0, 0, 1]]),
            },
            'arm_left': {
                'can': 'can0',
                'T_robot_to_arm': np.array([[1, 0, 0, 0.30],
                                               [0, 1, 0, 0.335],
                                               [0, 0, 1, 0.88],
                                               [0, 0, 0, 1]]),
            },
            'arm_right': {
                'can': 'can1',
                'T_robot_to_arm': np.array([[1, 0, 0, 0.30],
                                               [0, 1, 0, -0.335],
                                               [0, 0, 1, 0.92],
                                               [0, 0, 0, 1]]),
            },
        }
        for device_id in self.device_list:
            if 'rotation' in self.device_list[device_id]:
                rotation = np.eye(4)
                rotation[:3, :3] = self.device_list[device_id]['rotation']
                self.device_list[device_id]['T_robot_to_camera'] = self.device_list[device_id]['T_robot_to_camera'] @ rotation
        self.tracking_method = 'odom'
        self.tracking_method = 'T265'
        self.move_position_error_threshold = 0.1
        self.move_orientation_error_threshold = 0.3
        self.robot_pose_reset()

    def __call__(self, commond):
        action = commond[0]
        if action == 'move_forward':
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
        
        if hasattr(self, commond[0]):
            return getattr(self, commond[0])(commond)
        else:
            return 'Unsupported command.'

    def robot_pose_reset(self, commond=None):
        self.robot_pose_origin, timestamp = self.get_robot_pose_zero()
        return 'done'

    def get_rgbd_image(self, commond):
        camera_id = commond[1]
        if 'png' in commond or 'PNG' in commond:
            format = 'PNG'
        elif 'jpg' in commond or 'jpeg' in commond or 'JPG' in commond or 'JPEG' in commond:
            format = 'JPEG'
        else:
            format = 'array'
        serial_number = self.device_list[camera_id]['serial_number']
        self.init_device(camera_id)
        camera_param = None
        pose = None
        if 'pose' in commond:
            pose_list, timestamp_list = self.record_camera_pose(camera_id=camera_id, record_time=0.5)
            rgb_image, depth_image, timestamp = self.device_list[camera_id]['device'].capture_rgbd_image()
            closest_index = timestamp_match(timestamp_list, timestamp)
            pose = pose_list[closest_index]
            camera_param = self.device_list[camera_id]['device'].capture_camera_param()
        else:
            rgb_image, depth_image, timestamp = self.device_list[camera_id]['device'].capture_rgbd_image()
        if 'save' in commond:
            self.save_image(serial_number, rgb_image, depth_image, camera_param, pose, timestamp)
        if format in ['PNG', 'JPEG']:
            rgb_byte_io = io.BytesIO()
            if format == 'PNG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='PNG', optimize=True)
            elif format == 'JPEG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='JPEG', quality=90)
            rgb_image = rgb_byte_io
        if 'pose' in commond:
            if 'without_depth':
                return rgb_image, pose, timestamp
            else:
                return rgb_image, depth_image, pose, timestamp
        else:
            if 'without_depth':
                return rgb_image, timestamp
            else:
                return rgb_image, depth_image, timestamp

    def get_rgbd_image_rdt(self, commond):
        camera_id = commond[1]
        episode_index = commond[2]
        index = commond[3]
        serial_number = self.device_list[camera_id]['serial_number']
        self.init_device(camera_id)
        rgb_image, depth_image, timestamp = self.device_list[camera_id]['device'].capture_rgbd_image()
        if 'save' in commond:
            self.save_image_rdt(rgb_image, episode_index, index)
        return timestamp

    def get_pointcloud(self, commond):
        camera_id = commond[1]
        if 'FemtoBolt' in camera_id or '336L' in camera_id:
            self.init_device(camera_id)
            serial_number = self.device_list[camera_id]['serial_number']
            pose_list, timestamp_list = self.record_camera_pose(camera_id=camera_id, record_time=0.5)
            points, timestamp = self.device_list[camera_id]['device'].capture_color_point_cloud()
            closest_index = timestamp_match(timestamp_list, timestamp)
            pose = pose_list[closest_index]
            orientation = pose[:3, :3]
            position = pose[:3, 3]
            points[:, :3] = (orientation @ points[:, :3].T).T + position
            output_file_path_world = os.path.join(self.save_image_dir.format(serial_number), 'pc')
            if not os.path.exists(output_file_path_world):
                os.makedirs(output_file_path_world)
            output_file_path_world = os.path.join(output_file_path_world, f'{timestamp}.txt')
            np.savetxt(output_file_path_world, points, fmt='%.6f', delimiter=' ', header='X Y Z R G B')
            return 'done'
        else:
            'Device cannot support get_pointcloud.'

    def get_camera_intrinsic(self, commond):
        camera_id = commond[1]
        self.init_device(camera_id)
        color_intrinsics, depth_intrinsics = self.device_list[camera_id]['device'].capture_intrinsic()
        return color_intrinsics, depth_intrinsics

    def get_camera_extrinsic(self, commond):
        camera_id = commond[1]
        camera_position, camera_orientation, timestamp = self.get_camera_pose(camera_id)
        return camera_position, camera_orientation, timestamp

    def get_camera_xy_and_yaw(self, commond):
        camera_id = commond[1]
        robot_pose, timestamp = self.get_robot_pose()
        camera_pose = robot_pose @ self.device_list[camera_id]['T_robot_to_camera']
        camera_position = camera_pose[:3, 3]
        position_x, position_y = camera_position[0], camera_position[1]
        orientation = robot_pose[:3, :3]
        roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
        return position_x, position_y, yaw

    def get_robot_pose(self, commond=None):
        robot_pose, timestamp = self.get_robot_pose_zero()
        robot_pose = np.linalg.inv(self.robot_pose_origin) @ robot_pose
        return robot_pose, timestamp

    def robot_move(self, commond):
        self.goal = commond[1]
        while not self.reach_goal():
            self.robot_move_openloop(commond)
        return 'done'
    
    def get_arm_pose(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        arm_end_pose, arm_joint, arm_gripper_pose, timestamp = self.device_list[arm_id]['device'].get_arm_pose()
        if 'save' in commond:
            self.save_arm_pose(arm_id, arm_end_pose, arm_joint, arm_gripper_pose, timestamp)
        return arm_end_pose, arm_joint, arm_gripper_pose, timestamp
    
    def get_arm_pose_rdt(self, commond):
        arm_id = commond[1]
        episode_index = commond[2]
        index = commond[3]
        self.init_device(arm_id)
        arm_end_pose, arm_joint, arm_gripper_pose, timestamp = self.device_list[arm_id]['device'].get_arm_pose()
        if 'save' in commond:
            self.save_arm_pose_rdt(arm_end_pose, arm_joint, arm_gripper_pose, episode_index, index, timestamp)
        return arm_end_pose, arm_joint, arm_gripper_pose, timestamp

    def arm_reset(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].reset()
        return 'done'
    
    def arm_prepare(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].prepare()
        return 'done'
    
    def arm_open_gripper(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        arm_gripper_pose = 1
        self.device_list[arm_id]['device'].arm_gripper_ctrl(arm_gripper_pose)
        return 'done'
    
    def arm_close_gripper(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        arm_gripper_pose = 0
        self.device_list[arm_id]['device'].arm_gripper_ctrl(arm_gripper_pose)
        return 'done'
    
    def arm_enable(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].enable()
        return 'done'
    
    def arm_disable(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].disable()
        return 'done'

    def arm_move_camera(self, commond):
        camera_id = commond[1]
        self.init_device(camera_id)
        arm_id = commond[2]
        position = commond[3]
        if len(commond) >= 5:
            orientation = commond[4]
        else:
            orientation = None
        pose = np.eye(4)
        pose[:3, 3] = np.array(position)
        T_robot_to_camera = self.device_list[camera_id]['T_robot_to_camera']
        pose = T_robot_to_camera @ pose
        commond = (commond[0], commond[2], pose, orientation)
        self.arm_move_robot(commond)
        return 'done'

    def arm_move_robot(self, commond):
        arm_id = commond[1]
        pose = commond[2]
        if len(commond) >= 5:
            orientation = commond[3]
        else:
            orientation = None
        T_robot_to_arm = self.device_list[arm_id]['T_robot_to_arm']
        pose = np.linalg.inv(T_robot_to_arm) @ pose
        commond = (commond[0], commond[1], pose, orientation)
        self.arm_move_local(commond)
        return 'done'

    def arm_move_local(self, commond):
        arm_id = commond[1]
        self.init_device(arm_id)
        pose = commond[2]
        if len(commond) >= 5:
            orientation = commond[3]
        else:
            orientation = None
        # orientation = pose[:3, :3]
        position = pose[:3, 3] * 1000
        if orientation is None:
            arm_pose = [position[0], position[1], position[2] + 100, 0, 120, 0, 60]
        else:
            arm_pose = [position[0], position[1], position[2] + 100, orientation[0], orientation[1], orientation[2], orientation[3]]
        self.device_list[arm_id]['device'].arm_end_pose_ctrl(arm_pose)
        return 'done'

    def arm_end_pose_ctrl(self, commond):
        arm_id = commond[1]
        arm_end_pose = commond[2]
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].arm_end_pose_ctrl(arm_end_pose)
        return 'done'

    def arm_joint_ctrl(self, commond):
        arm_id = commond[1]
        arm_joint = commond[2]
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].arm_joint_ctrl(arm_joint)
        return 'done'

    def init_device(self, device_id):
        if 'device' not in self.device_list[device_id]:
            if 'D435i' in device_id:
                serial_number = self.device_list[device_id]['serial_number']
                self.device_list[device_id]['device'] = D435i(serial_number)
            elif 'T265' in device_id:
                serial_number = self.device_list[device_id]['serial_number']
                self.device_list[device_id]['device'] = T265(serial_number)
            elif 'FemtoBolt' in device_id or '336L' in device_id:
                serial_number = self.device_list[device_id]['serial_number']
                self.device_list[device_id]['device'] = OrbbecCamera(serial_number)
            elif 'arm' in device_id:
                can = self.device_list[device_id]['can']
                self.device_list[device_id]['device'] = PiperArm(can)

    def get_tracking_pose(self, commond=None):
        self.init_device(self.tracking_method)
        position, orientation, timestamp = self.device_list[self.tracking_method]['device'].get_pose()
        orientation = R.from_quat(quat_wxyz_to_xyzw(orientation)).as_matrix()
        T = np.eye(4)
        T[:3, :3] = orientation
        T[:3, 3] = position
        return T, timestamp

    def get_robot_pose_zero(self, commond=None):
        if self.tracking_method == 'odom':
            robot_pose, timestamp = get_odom_pose()
        elif self.tracking_method == 'T265':
            T_tracking, timestamp = self.get_tracking_pose()
            T_robot_to_camera = self.device_list[self.tracking_method]['T_robot_to_camera']
            robot_pose = T_robot_to_camera @ T_tracking @ np.linalg.inv(T_robot_to_camera)
        return robot_pose, timestamp

    def robot_move_openloop(self, commond):
        self.goal = commond[1]
        robot_pose, timestamp = self.get_robot_pose()
        T = np.linalg.inv(robot_pose) @ self.goal
        orientation = T[:3, :3]
        position = T[:2, 3]
        roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
        path_length = np.linalg.norm(position)
        
        if path_length < 1:
            self.robot_move_translation(position)
            # self.robot_move_rotation(yaw)
        else:
            x, y = position
            rotation_0 = np.arctan2(y, x)
            self.robot_move_rotation(rotation_0)
            # position = np.array([path_length, 0])
            robot_pose, timestamp = self.get_robot_pose()
            T = np.linalg.inv(robot_pose) @ self.goal
            orientation = T[:3, :3]
            position = T[:2, 3]
            roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
            self.robot_move_translation(position)
            # roll, pitch, rotation_1 = (R.from_euler('xyz', (0, 0, rotation_0)).inv() * R.from_euler('xyz', (0, 0, yaw))).as_euler('xyz')
            # self.robot_move_rotation(yaw)
        return 'done'
    
    def robot_move_translation(self, position):
        x, y = position
        x = x / 1.0 * 1.0
        y = y / 1.0 * 1.0
        speed = 0.001
        distance = np.linalg.norm(position)
        t = int(distance / speed)
        speed_x = 0.1 * x / distance
        speed_y = 0.1 * y / distance
        if t > 0:
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: ' + str(speed_x) + ', y: ' + str(speed_y) + ', z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 100 -t ' + str(t))
        return 'done'

    def robot_move_rotation(self, yaw):
        yaw = yaw / np.pi * 180
        t = abs(int(yaw / 90 * 800))
        if t > 0:
            if yaw > 0:
                os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}" -r 100 -t ' + str(t))
            elif yaw < 0:
                os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.2}}" -r 100 -t ' + str(t))
        return 'done'
    
    def reach_goal(self):
        robot_pose, timestamp = self.get_robot_pose()
        goal = self.goal
        error = np.linalg.inv(goal) @ robot_pose
        move_position_error = np.linalg.norm(error[:2, 3])
        move_orientation_error = R.from_matrix(error[:3, :3]).as_euler('xyz')[2]
        reach_goal = move_position_error <= self.move_position_error_threshold and move_orientation_error <= self.move_orientation_error_threshold
        return reach_goal

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
            # extrinsic = np.eye(4)
            # extrinsic[:3, :3] = R.from_quat(quat_wxyz_to_xyzw(pose[1])).as_matrix()
            # extrinsic[:3, 3] = pose[0]
            # np.savetxt(pose_path, extrinsic)
            np.savetxt(pose_path, pose)

    def save_image_rdt(self, rgb_image, episode_index, index):
        rgb_image_dir = self.save_image_dir.format('arm_right', episode_index)
        if not os.path.exists(rgb_image_dir):
            os.makedirs(rgb_image_dir)
        rgb_image_path = os.path.join(rgb_image_dir, f'{index:0>6}.png')
        Image.fromarray(rgb_image).save(rgb_image_path)

    def save_arm_pose(self, arm_id, arm_end_pose, arm_joint, arm_gripper_pose, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        arm_pose_path = os.path.join(self.save_arm_pose_dir.format(arm_id), 'pose', f'{timestamp}.npy')
        arm_pose = {
            'arm_end_pose': arm_end_pose,
            'arm_joint': arm_joint,
            'arm_gripper_pose': arm_gripper_pose,
        }
        pickle.dump(arm_pose, open(arm_pose_path, 'wb'))

    def save_arm_pose_rdt(self, arm_end_pose, arm_joint, arm_gripper_pose, episode_index, index, timestamp=None):
        arm_pose_dir = self.save_arm_pose_dir.format('arm_right', episode_index)
        if not os.path.exists(arm_pose_dir):
            os.makedirs(arm_pose_dir)
        arm_pose_path = os.path.join(arm_pose_dir, f'{index:0>6}.pkl')
        arm_pose = {
            'arm_end_pose': arm_end_pose,
            'arm_joint': arm_joint,
            'arm_gripper_pose': arm_gripper_pose,
            'timestamp': timestamp,
        }
        pickle.dump(arm_pose, open(arm_pose_path, 'wb'))

    def get_robot_xy_and_yaw(self):
        # position, orientation, timestamp = self.get_robot_pose()
        robot_pose, timestamp = self.get_robot_pose()
        orientation = robot_pose[:3, :3]
        position = robot_pose[:3, 3]
        # position, orientation, timestamp = self.get_robot_pose()
        robot_pose, timestamp = self.get_robot_pose()
        orientation = robot_pose[:3, :3]
        position = robot_pose[:3, 3]
        position_x, position_y = position[0], position[1]
        roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
        roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
        return position_x, position_y, yaw
    
    def get_camera_pose(self, camera_id):
        robot_pose, timestamp = self.get_robot_pose()
        camera_pose = robot_pose @ self.device_list[camera_id]['T_robot_to_camera']
        return camera_pose, timestamp
    
    def record_camera_pose(self, camera_id, record_time=1.0):
        pose_list = []
        timestamp_list = []
        self.init_device(self.tracking_method)
        tracking_fps = self.device_list[self.tracking_method]['device'].get_fps()
        num_record_frame = int(tracking_fps * record_time)
        for i in range(num_record_frame):
            camera_pose, timestamp = self.get_camera_pose(camera_id)
            pose_list.append(camera_pose)
            timestamp_list.append(timestamp)
        return pose_list, timestamp_list
    
    