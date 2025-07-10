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
from .base_odom import get_odom_pose, get_odom_xy_and_yaw, get_camera_xy_and_yaw
from .point_cloud_generator import generate_point_cloud
from .config import save_image_dir, save_rdt_image_dir, save_rdt_arm_pose_dir, device_list


class Hexmove():
    def __init__(self) -> None:
        self.save_image_dir = save_image_dir
        self.save_rdt_image_dir = save_rdt_image_dir
        self.save_rdt_arm_pose_dir = save_rdt_arm_pose_dir
        self.device_list = device_list
        self.tracking_method = 'odom'
        self.tracking_method = 'MASt3R-SLAM'
        self.tracking_method = 'T265'
        self.move_position_error_threshold = 0.1
        self.move_orientation_error_threshold = 0.3
        self.robot_pose_reset_done = False
        self.robot_pose_reset()

        # self.init_device("FemtoBolt_down")
        self.init_device("FemtoBolt_up")
        # self.init_device("336L_head")

        # self.init_device("336L_arm_right")
        # self.init_device("336L_arm_left")
        
        
    def base_action(self, action):
        if action == 'move_forward':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1')
            return 'move_forward done'
        elif action == 'move_backward':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1')
            return 'move_backward done'
        elif action == 'move_left depth':
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

    def robot_pose_reset(self):
        self.robot_pose_origin, timestamp = self.get_robot_pose_zero()
        self.robot_pose_reset_done = True
        return 'done'

    def test_sleep(self):
        for i in range(10):
            time.sleep(1)
            print(f"sleep {i} seconds")
        return 'done'

    def get_rgbd_image(self, camera_id=None, pose=False, format='array', save=False, pose_type='extrinsic', without_depth=False):
        serial_number = self.device_list[camera_id]['serial_number']
        self.init_device(camera_id)
        camera_param = None
        pose = None
        if pose:
            pose_list, timestamp_list = self.record_camera_pose(camera_id=camera_id, record_time=0.5, pose_type=pose_type)
            rgb_image, depth_image, timestamp = self.device_list[camera_id]['device'].capture_rgbd_image()
            closest_index = timestamp_match(timestamp_list, timestamp)
            pose = pose_list[closest_index]
            camera_param = self.device_list[camera_id]['device'].capture_camera_param()
        else:
            rgb_image, depth_image, timestamp = self.device_list[camera_id]['device'].capture_rgbd_image()
        if save:
            self.save_image(serial_number, rgb_image, depth_image, camera_param, pose, timestamp)
        if format in ['PNG', 'JPEG']:
            rgb_byte_io = io.BytesIO()
            if format == 'PNG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='PNG', optimize=True)
            elif format == 'JPEG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='JPEG', quality=90)
            rgb_image = rgb_byte_io
        if pose:
            if without_depth:
                return rgb_image, pose, timestamp
            else:
                return rgb_image, depth_image, pose, timestamp
        else:
            if without_depth:
                return rgb_image, timestamp
            else:
                return rgb_image, depth_image, timestamp

    def get_rgb_image_rdt(self, camera_id, episode_index=None, index=None, position=None, format='array', save=False, save_dir=None):
        # self.init_device(camera_id)
        rgb_image, timestamp = self.device_list[camera_id]['device'].capture_rgb_image()

        if format in ['PNG', 'JPEG']:
            rgb_byte_io = io.BytesIO()
            if format == 'PNG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='PNG', optimize=True)
            elif format == 'JPEG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='JPEG', quality=90)
            rgb_image = rgb_byte_io

        if save:
            self.save_image_rdt(rgb_image, position, episode_index, index, save_dir=save_dir)
            return timestamp
        else:
            return rgb_image, timestamp

    def get_rgbd_image_rdt(self, camera_id, episode_index=None, index=None, position=None, format='array', save=False, save_dir=None):
        # self.init_device(camera_id)
        rgb_image, depth_image, timestamp = self.device_list[camera_id]['device'].capture_rgbd_image()
        camera_param = self.device_list[camera_id]['device'].capture_camera_param()
        
        # Get camera extrinsic parameters
        pose_list, timestamp_list = self.record_camera_pose(camera_id=camera_id, record_time=0.5, pose_type='extrinsic')
        closest_index = timestamp_match(timestamp_list, timestamp)
        pose = pose_list[closest_index]
        
        start_time = time.time()
        # Direct point cloud calculation from RGB and depth images
        # Extract camera intrinsics
        fx = camera_param.rgb_intrinsic.fx
        fy = camera_param.rgb_intrinsic.fy
        cx = camera_param.rgb_intrinsic.cx
        cy = camera_param.rgb_intrinsic.cy
        print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        depth_scale = 0.001  # Convert from millimeters to meters
        
        # Create a grid of pixel coordinates
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth image to meters
        z = depth_image.astype(np.float32) * depth_scale
        
        # Filter out invalid depth values
        valid_mask = (z > 0) & (z < 10)  # Only keep points within reasonable distance
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]
        
        # Calculate 3D coordinates
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        
        # Get RGB values for valid points
        colors = rgb_image[v_valid, u_valid]
        
        # Combine points and colors
        points = np.column_stack((x, y, z_valid))
        point_cloud = np.hstack((points, colors))
        
        end_time = time.time()
        print(f"Generation time: {end_time - start_time} seconds")

        start_time = time.time()
        # Grid sampling for point cloud
        def grid_sample_pcd(point_cloud, grid_size=0.005):
            """
            A simple grid sampling function for point clouds.

            Parameters:
            - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                        The first 3 columns represent the coordinates (x, y, z).
                        The next 3 columns (if present) can represent additional attributes like color or normals.
            - grid_size: Size of the grid for sampling.

            Returns:
            - A NumPy array of sampled points with the same shape as the input but with fewer rows.
            """
            coords = point_cloud[:, :3]  # Extract coordinates
            scaled_coords = coords / grid_size
            grid_coords = np.floor(scaled_coords).astype(int)
            
            # Create unique grid keys
            keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
            
            # Select unique points based on grid keys
            _, indices = np.unique(keys, return_index=True)
            
            # Return sampled points
            return point_cloud[indices]
        
        # Apply grid sampling instead of random sampling
        if point_cloud.shape[0] > 4096:
            point_cloud = grid_sample_pcd(point_cloud, grid_size=0.01)  # Adjust grid_size as needed
            print(f"sampled point cloud shape: {point_cloud.shape}")
            # If still too many points after grid sampling, subsample further
            if point_cloud.shape[0] > 4096:
                point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], 4096, replace=False), :]
        
        # print cloud shape
        print(f"Point cloud shape after doublesampling: {point_cloud.shape}")
        end_time = time.time()
        print(f"Sampling time: {end_time - start_time} seconds")
        
        if format in ['PNG', 'JPEG']:
            rgb_byte_io = io.BytesIO()
            if format == 'PNG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='PNG', optimize=True)
            elif format == 'JPEG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='JPEG', quality=90)
            rgb_image_formatted = rgb_byte_io
        else:
            rgb_image_formatted = rgb_image

        if save:
            start_time = time.time()
            
            # Create directory structure with ext folder
            if save_dir is None:
                base_dir = os.path.dirname(os.path.dirname(self.save_rdt_image_dir.format('arm_right', episode_index, position)))
                ext_dir = os.path.join(base_dir, 'ext')
                pc_dir = os.path.join(base_dir, 'pointcloud')
                depth_dir = os.path.join(base_dir, 'depth')
                # pose_dir = os.path.join(base_dir, 'pose')
            else:
                base_dir = os.path.dirname(os.path.dirname(self.save_rdt_image_dir.format(save_dir, episode_index, position)))
                ext_dir = os.path.join(base_dir, 'ext')
                pc_dir = os.path.join(base_dir, 'pointcloud')
                depth_dir = os.path.join(base_dir, 'depth')
                # pose_dir = os.path.join(base_dir, 'pose')
            
            # Create directories
            os.makedirs(ext_dir, exist_ok=True)
            os.makedirs(pc_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            # os.makedirs(pose_dir, exist_ok=True)
            
            # Save RGB image with sequential naming
            rgb_image_path = os.path.join(ext_dir, f'{index:0>6}.jpg')
            Image.fromarray(rgb_image).save(rgb_image_path)
            print(f"RGB image saved at {rgb_image_path}")
            
            # Save depth image with sequential naming
            depth_image_path = os.path.join(depth_dir, f'{index:0>6}.png')
            Image.fromarray(depth_image, mode='I;16').save(depth_image_path)
            print(f"Depth image saved at {depth_image_path}")

            # # Save camera pose with sequential naming
            # pose_path = os.path.join(pose_dir, f'{index:0>6}.pkl')
            # with open(pose_path, 'wb') as f:
            #     pickle.dump(pose, f)
            # print(f"Pose saved at {pose_path}")

            # Save point cloud with sequential naming
            pc_path = os.path.join(pc_dir, f'{index:0>6}.txt')
            np.savetxt(pc_path, point_cloud, fmt='%.6f', delimiter=' ', header='X Y Z R G B')
            end_time = time.time()
            print(f"Point cloud saved at {pc_path} in {end_time - start_time:.2f} seconds.")
            return timestamp
        else:
            return rgb_image_formatted, point_cloud, timestamp

    def get_pointcloud(self, camera_id):
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
            os.makedirs(output_file_path_world, exist_ok=True)
            output_file_path_world = os.path.join(output_file_path_world, f'{timestamp}.txt')
            np.savetxt(output_file_path_world, points, fmt='%.6f', delimiter=' ', header='X Y Z R G B')
            return points
        else:
            return 'Device cannot support get_pointcloud.'

    def get_camera_intrinsic(self, camera_id):
        self.init_device(camera_id)
        color_intrinsics, depth_intrinsics = self.device_list[camera_id]['device'].capture_intrinsic()
        return color_intrinsics, depth_intrinsics

    def get_camera_extrinsic(self, camera_id):
        camera_pose, timestamp = self.get_camera_pose(camera_id, pose_type='extrinsic')
        camera_extrinsic = camera_pose
        return camera_extrinsic, timestamp

    def get_camera_xy_and_yaw(self, camera_id):
        camera_pose, timestamp = self.get_camera_pose(camera_id, pose_type='xy_and_yaw')
        position_x, position_y, yaw = camera_pose
        return position_x, position_y, yaw

    def get_robot_pose(self):
        if not self.robot_pose_reset_done:
            self.robot_pose_reset()
        robot_pose, timestamp = self.get_robot_pose_zero()
        robot_pose = np.linalg.inv(self.robot_pose_origin) @ robot_pose
        return robot_pose, timestamp

    def robot_move(self, goal):
        self.goal = goal
        while not self.reach_goal():
            self.robot_move_openloop(goal)
        return 'done'
    
    def get_arm_pose(self, arm_id, save=False):
        self.init_device(arm_id)
        arm_end_pose, arm_joint, arm_gripper_pose, timestamp = self.device_list[arm_id]['device'].get_arm_pose()
        if save:
            self.save_arm_pose(arm_id, arm_end_pose, arm_joint, arm_gripper_pose, timestamp)
        return arm_end_pose, arm_joint, arm_gripper_pose, timestamp
    
    def get_arm_pose_rdt(self, arm_id, episode_index, index, save=False):
        self.init_device(arm_id)
        arm_end_pose, arm_joint, arm_gripper_pose, timestamp = self.device_list[arm_id]['device'].get_arm_pose()
        if save:
            self.save_arm_pose_rdt(arm_end_pose, arm_joint, arm_gripper_pose, episode_index, index, timestamp)
        return arm_end_pose, arm_joint, arm_gripper_pose, timestamp

    def get_arm_pose_idp3(self, arm_id, episode_index, index, save=False, save_dir=None):
        self.init_device(arm_id)
        self.init_device("arm_left")
        right_arm_end_pose, right_arm_joint, right_arm_gripper_pose, right_timestamp = self.device_list[arm_id]['device'].get_arm_pose()
        left_arm_end_pose, left_arm_joint, left_arm_gripper_pose, left_timestamp = self.device_list["arm_left"]['device'].get_arm_pose()
        arm_end_pose = np.concatenate((left_arm_end_pose, right_arm_end_pose), axis=0)
        arm_joint = np.concatenate((left_arm_joint, right_arm_joint), axis=0)
        arm_gripper_pose = np.concatenate((left_arm_gripper_pose, right_arm_gripper_pose), axis=0)

        if save:
            self.save_arm_pose_idp3(arm_end_pose, arm_joint, arm_gripper_pose, episode_index, index, left_timestamp, save_dir=save_dir)
        return arm_end_pose, arm_joint, arm_gripper_pose, left_timestamp

    def arm_reset(self, arm_id):
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].reset()
        return 'done'
    
    def arm_prepare(self, arm_id):
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].prepare()
        return 'done'
    
    def arm_open_gripper(self, arm_id):
        self.init_device(arm_id)
        arm_gripper_pose = 1
        self.device_list[arm_id]['device'].arm_gripper_ctrl(arm_gripper_pose)
        return 'done'
    
    def arm_close_gripper(self, arm_id):
        self.init_device(arm_id)
        arm_gripper_pose = 0.42
        self.device_list[arm_id]['device'].arm_gripper_ctrl(arm_gripper_pose)
        return 'done'
    
    def arm_enable(self, arm_id):
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].enable()
        return 'done'
    
    def arm_disable(self, arm_id):
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].disable()
        return 'done'

    def arm_move_camera(self, camera_id, arm_id, position, orientation=None):
        self.init_device(camera_id)
        pose = np.eye(4)
        pose[:3, 3] = np.array(position)
        T_camera_to_robot = self.device_list[camera_id]['T_camera_to_robot']
        pose = T_camera_to_robot @ pose
        self.arm_move_robot(arm_id, pose, orientation)
        return 'done'

    def arm_move_robot(self, arm_id, pose, orientation=None):
        T_arm_to_robot = self.device_list[arm_id]['T_arm_to_robot']
        pose = np.linalg.inv(T_arm_to_robot) @ pose
        self.arm_move_local(arm_id, pose, orientation)
        return 'done'

    def arm_move_local(self, arm_id, position, orientation=None):
        self.init_device(arm_id)
        # orientation = pose[:3, :3]
        # position = pose[:3, 3] * 1000
        # position = pose * 1000
        if orientation is None:
            arm_pose = [position[0], position[1], position[2], 0, 120, 0, 60]
        else:
            arm_pose = [position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3]]
        self.device_list[arm_id]['device'].arm_end_pose_ctrl(arm_pose)
        return 'done'

    def arm_end_pose_ctrl(self, arm_id, arm_end_pose):
        self.init_device(arm_id)
        self.device_list[arm_id]['device'].arm_end_pose_ctrl(arm_end_pose)
        return 'done'

    def arm_joint_ctrl(self, arm_id, arm_joint):
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

    def get_tracking_pose(self):
        self.init_device(self.tracking_method)
        position, orientation, timestamp = self.device_list[self.tracking_method]['device'].get_pose()
        orientation = R.from_quat(quat_wxyz_to_xyzw(orientation)).as_matrix()
        tracking_pose = np.eye(4)
        tracking_pose[:3, :3] = orientation
        tracking_pose[:3, 3] = position
        return tracking_pose, timestamp

    def get_robot_pose_zero(self):
        if self.tracking_method == 'odom':
            robot_pose, timestamp = get_odom_pose()
        elif self.tracking_method == 'T265':
            tracking_pose, timestamp = self.get_tracking_pose()
            T_camera_to_robot = self.device_list[self.tracking_method]['T_camera_to_robot']
            robot_pose = T_camera_to_robot @ tracking_pose @ np.linalg.inv(T_camera_to_robot)
        return robot_pose, timestamp

    def robot_move_openloop(self, goal):
        self.goal = goal
        robot_pose, timestamp = self.get_robot_pose()
        T = np.linalg.inv(robot_pose) @ self.goal
        orientation = T[:3, :3]
        position = T[:2, 3]
        roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
        path_length = np.linalg.norm(position)
        
        if path_length < 0.2:
            self.robot_move_translation(position)
            self.robot_move_rotation(yaw)
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
            self.robot_move_rotation(yaw)
        return 'done'
    
    def robot_move_translation(self, position):
        x, y = position
        x = x / 1.0 * 1.0
        y = y / 1.0 * 1.0
        # speed = 0.0035
        speed = 0.001095
        speed = 0.0009
        distance = np.linalg.norm(position)
        t = int(distance / speed)
        # speed_x = 0.1 * speed * x / distance
        # speed_y = 0.1 * speed * y / distance
        speed_x = 0.1 * x / distance
        speed_y = 0.1 * y / distance
        if t > 0:
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: ' + str(speed_x) + ', y: ' + str(speed_y) + ', z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 100 -t ' + str(t))
        return 'done'

    def robot_move_rotation(self, yaw):
        yaw = yaw / np.pi * 180
        t = abs(int(yaw / 90 * 850))
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
            np.savetxt(pose_path, pose)

    def save_image_rdt(self, rgb_image, position, episode_index, index, save_dir=None):
        if save_dir is None:
            rgb_image_dir = self.save_rdt_image_dir.format('arm_right', episode_index, position)
        else:
            rgb_image_dir = self.save_rdt_image_dir.format(save_dir, episode_index, position)
        os.makedirs(rgb_image_dir, exist_ok=True)
        rgb_image_path = os.path.join(rgb_image_dir, f'{index:0>6}.jpg')
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

    def save_arm_pose_rdt(self, arm_end_pose, arm_joint, arm_gripper_pose, episode_index, index, timestamp=None, save_dir=None):
        arm_pose_dir = self.save_rdt_arm_pose_dir.format('arm_right', episode_index)
        os.makedirs(arm_pose_dir, exist_ok=True)
        arm_pose_path = os.path.join(arm_pose_dir, f'{index:0>6}.pkl')
        arm_pose = {
            'arm_end_pose': arm_end_pose,
            'arm_joint': arm_joint,
            'arm_gripper_pose': arm_gripper_pose,
            'timestamp': timestamp,
        }
        pickle.dump(arm_pose, open(arm_pose_path, 'wb'))

    def save_arm_pose_idp3(self, arm_end_pose, arm_joint, arm_gripper_pose, episode_index, index, timestamp=None, save_dir=None):
        if save_dir is None:
            arm_pose_dir = self.save_rdt_arm_pose_dir.format("arm_right", episode_index)
        else:
            arm_pose_dir = self.save_rdt_arm_pose_dir.format(save_dir, episode_index)
        os.makedirs(arm_pose_dir, exist_ok=True)
        print(arm_pose_dir)
        arm_pose_path = os.path.join(arm_pose_dir, f'{index:0>6}.pkl')
        arm_pose = {
            'arm_end_pose': arm_end_pose,
            'arm_joint': arm_joint,
            'arm_gripper_pose': arm_gripper_pose,
            'timestamp': timestamp,
        }
        pickle.dump(arm_pose, open(arm_pose_path, 'wb'))

    def get_robot_xy_and_yaw(self):
        robot_pose, timestamp = self.get_robot_pose()
        orientation = robot_pose[:3, :3]
        position = robot_pose[:3, 3]
        position_x, position_y = position[0], position[1]
        roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
        return position_x, position_y, yaw
    
    def get_camera_pose(self, camera_id, pose_type='extrinsic'):
        robot_pose, timestamp = self.get_robot_pose()
        camera_pose = robot_pose @ self.device_list[camera_id]['T_camera_to_robot']
        if pose_type == 'extrinsic':
            return camera_pose, timestamp
        if pose_type == 'xy_and_yaw':
            camera_position = camera_pose[:3, 3]
            position_x, position_y = camera_position[0], camera_position[1]
            orientation = robot_pose[:3, :3]
            roll, pitch, yaw = R.from_matrix(orientation).as_euler('xyz')
            return (position_x, position_y, yaw), timestamp
    
    def record_camera_pose(self, camera_id, record_time=1.0, pose_type='extrinsic'):
        pose_list = []
        timestamp_list = []
        self.init_device(self.tracking_method)
        tracking_fps = self.device_list[self.tracking_method]['device'].get_fps()
        num_record_frame = int(tracking_fps * record_time)
        for i in range(num_record_frame):
            camera_pose, timestamp = self.get_camera_pose(camera_id, pose_type)
            pose_list.append(camera_pose)
            timestamp_list.append(timestamp)
        return pose_list, timestamp_list
    
    def get_dual_arm_idp3(self, arm_id):
        self.init_device(arm_id)
        self.init_device("arm_left")
        right_arm_end_pose, right_arm_joint, right_arm_gripper_pose, right_timestamp = self.device_list[arm_id]['device'].get_arm_pose()
        left_arm_end_pose, left_arm_joint, left_arm_gripper_pose, left_timestamp = self.device_list["arm_left"]['device'].get_arm_pose()
        arm_end_pose = np.concatenate((left_arm_end_pose, right_arm_end_pose), axis=0)
        arm_joint = np.concatenate((left_arm_joint, right_arm_joint), axis=0)
        arm_gripper_pose = np.concatenate((left_arm_gripper_pose, right_arm_gripper_pose), axis=0)
        return arm_end_pose, arm_joint, arm_gripper_pose, left_timestamp
    
    def get_rgbd_image_fast(self, camera_id, episode_index=None, index=None, position=None, format='array', save=False, save_dir=None):
        rgb_image, depth_image, timestamp = self.device_list[camera_id]['device'].capture_rgbd_image()
        camera_param = self.device_list[camera_id]['device'].capture_camera_param()
        
        # # Get camera extrinsic parameters
        # pose_list, timestamp_list = self.record_camera_pose(camera_id=camera_id, record_time=0.5, pose_type='extrinsic')
        # closest_index = timestamp_match(timestamp_list, timestamp)
        # pose = pose_list[closest_index]
        
        # start_time = time.time()
        # # Direct point cloud calculation from RGB and depth images
        # # Extract camera intrinsics
        # fx = camera_param.rgb_intrinsic.fx
        # fy = camera_param.rgb_intrinsic.fy
        # cx = camera_param.rgb_intrinsic.cx
        # cy = camera_param.rgb_intrinsic.cy
        # print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        # depth_scale = 0.001  # Convert from millimeters to meters
        
        # # Create a grid of pixel coordinates
        # height, width = depth_image.shape
        # u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # # Convert depth image to meters
        # z = depth_image.astype(np.float32) * depth_scale
        
        # # Filter out invalid depth values
        # valid_mask = (z > 0) & (z < 10)  # Only keep points within reasonable distance
        # u_valid = u[valid_mask]
        # v_valid = v[valid_mask]
        # z_valid = z[valid_mask]
        
        # # Calculate 3D coordinates
        # x = (u_valid - cx) * z_valid / fx
        # y = (v_valid - cy) * z_valid / fy
        
        # # Get RGB values for valid points
        # colors = rgb_image[v_valid, u_valid]
        
        # # Combine points and colors
        # points = np.column_stack((x, y, z_valid))
        # point_cloud = np.hstack((points, colors))
        
        # end_time = time.time()
        # print(f"Generation time: {end_time - start_time} seconds")

        # start_time = time.time()
        # # Grid sampling for point cloud
        # def grid_sample_pcd(point_cloud, grid_size=0.005):
        #     """
        #     A simple grid sampling function for point clouds.

        #     Parameters:
        #     - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
        #                 The first 3 columns represent the coordinates (x, y, z).
        #                 The next 3 columns (if present) can represent additional attributes like color or normals.
        #     - grid_size: Size of the grid for sampling.

        #     Returns:
        #     - A NumPy array of sampled points with the same shape as the input but with fewer rows.
        #     """
        #     coords = point_cloud[:, :3]  # Extract coordinates
        #     scaled_coords = coords / grid_size
        #     grid_coords = np.floor(scaled_coords).astype(int)
            
        #     # Create unique grid keys
        #     keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
            
        #     # Select unique points based on grid keys
        #     _, indices = np.unique(keys, return_index=True)
            
        #     # Return sampled points
        #     return point_cloud[indices]
            
        # # Apply grid sampling instead of random sampling
        # if point_cloud.shape[0] > 4096:
        #     point_cloud = grid_sample_pcd(point_cloud, grid_size=0.01)  # Adjust grid_size as needed
        #     print(f"sampled point cloud shape: {point_cloud.shape}")
        #     # If still too many points after grid sampling, subsample further
        #     if point_cloud.shape[0] > 4096:
        #         point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], 4096, replace=False), :]
        
        # # print cloud shape
        # print(f"Point cloud shape after doublesampling: {point_cloud.shape}")
        # end_time = time.time()
        # print(f"Sampling time: {end_time - start_time} seconds")
        
        if format in ['PNG', 'JPEG']:
            rgb_byte_io = io.BytesIO()
            if format == 'PNG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='PNG', optimize=True)
            elif format == 'JPEG':
                Image.fromarray(rgb_image).save(rgb_byte_io, format='JPEG', quality=90)
            rgb_image_formatted = rgb_byte_io
        else:
            rgb_image_formatted = rgb_image
        if save:
            if save_dir is None:
                base_dir = os.path.dirname(os.path.dirname(self.save_rdt_image_dir.format('arm_right', episode_index, position)))
                ext_dir = os.path.join(base_dir, 'ext')
                # pc_dir = os.path.join(base_dir, 'pointcloud')
                depth_dir = os.path.join(base_dir, 'depth')
                # pose_dir = os.path.join(base_dir, 'pose')
            else:
                base_dir = os.path.dirname(os.path.dirname(self.save_rdt_image_dir.format(save_dir, episode_index, position)))
                ext_dir = os.path.join(base_dir, 'ext')
                # pc_dir = os.path.join(base_dir, 'pointcloud')
                depth_dir = os.path.join(base_dir, 'depth')
                # pose_dir = os.path.join(base_dir, 'pose')
            
            # Create directories
            os.makedirs(ext_dir, exist_ok=True)
            # os.makedirs(pc_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            # os.makedirs(pose_dir, exist_ok=True)
            
            # Save RGB image with sequential naming
            rgb_image_path = os.path.join(ext_dir, f'{index:0>6}.jpg')
            Image.fromarray(rgb_image).save(rgb_image_path)
            print(f"RGB image saved at {rgb_image_path}")
            
            # Save depth image with sequential naming
            depth_image_path = os.path.join(depth_dir, f'{index:0>6}.png')
            Image.fromarray(depth_image, mode='I;16').save(depth_image_path)
            print(f"Depth image saved at {depth_image_path}")

            # # Save camera pose with sequential naming
            # pose_path = os.path.join(pose_dir, f'{index:0>6}.pkl')
            # with open(pose_path, 'wb') as f:
            #     pickle.dump(pose, f)
            # print(f"Pose saved at {pose_path}")

            # Save point cloud with sequential naming
            # pc_path = os.path.join(pc_dir, f'{index:0>6}.txt')
            # np.savetxt(pc_path, point_cloud, fmt='%.6f', delimiter=' ', header='X Y Z R G B')
            # end_time = time.time()
            # print(f"Point cloud saved at {pc_path} in {end_time - start_time:.2f} seconds.")
            
            return timestamp
        else:
            # return rgb_image_formatted, point_cloud, timestamp
            return