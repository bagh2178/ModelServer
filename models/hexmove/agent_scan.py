import os
import time
import math
from abc import ABC

import cv2
import rclpy
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from scipy.spatial.transform import Rotation as R
from utils import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from odom_subscriber import get_odom_pose, get_odom_xy_and_yaw, get_camera_xy_and_yaw
from orbbec import FemtoBolt

# import torch
# import torch.nn as nn
from dataclasses import dataclass
from datetime import datetime


class Agent(ABC):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.args = args
        self.save_image_dir = args.save_image_dir
        self.camera_list = {
            'D435i_top': {
                'serial_number': '337322070914',
                'R_camera_to_robot': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                'translation_camera_to_robot': np.array([0.25, 0, 1.3]),
            },
            'D435i_down': {
                'serial_number': '327122078142',
                'R_camera_to_robot': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                'translation_camera_to_robot': np.array([0.3, 0, 0.88]),
            },
            'T265': {
                'serial_number': '119622110447',
                'R_camera_to_robot': np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                'translation_camera_to_robot': np.array([0.35, 0, 0]),
            },
            'FemtoBolt_down': {
                'serial_number': 'CL8M841006W',
                'R_camera_to_robot': np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                'translation_camera_to_robot': np.array([0.43, 0, 0.88]),
            },
        }
        self.tracking_method = args.tracking_method

        # init robot
        self.robot_position = None
        self.robot_orientation = None
        self.step_number = 0
        self.t265_timestep = None
        self.rgb_timestep = None

        # add t265 camera
        self.t265_pipeline = rs.pipeline()
        self.t265_config = rs.config()
        if args.t265_serial_number:
            self.t265_config.enable_device(args.t265_serial_number)
        self.t265_config.enable_stream(rs.stream.pose)
        # Start pipeline
        profile = self.t265_pipeline.start(self.t265_config)

        # add rgb-d camera
        self.rgbd_sensor = FemtoBolt(args.rgbd_serial_number)

    def exec_action(self, commond):
        action = commond[0]
        if action == 'get_intrinsic':
            camera_id = commond[1]
            self.init_camera(camera_id)
            color_intrinsics, depth_intrinsics = self.camera_list[camera_id]['camera'].capture_intrinsic()
            return color_intrinsics, depth_intrinsics
        # elif action == 'get_rgbd_frame':
        #     camera_id = commond[1]
        #     serial_number = self.camera_list[camera_id]['serial_number']
        #     self.init_camera(camera_id)
        #     rgb_image, depth_image = self.camera_list[camera_id]['camera'].capture_rgbd_image()
        #     x, y, yaw = self.get_camera_xy_and_yaw(camera_offset=0.3)
        #     # self.save_image(serial_number, rgb_image, depth_image)
        #     return rgb_image, depth_image, (x, y, yaw)
        elif action == 'get_tracking_pose':
            position, orientation, timestep = self.get_tracking_pose()
            return position, orientation
        # elif action == 'get_robot_pose':
        #     position, orientation = self.get_robot_pose()
        #     return position, orientation
        elif action == 'get_robot_xy_and_yaw':
            x, y, yaw = self.get_robot_xy_and_yaw()
            return x, y, yaw
        elif action == 'get_camera_pose':
            camera_id = commond[1]
            camera_pose, camera_orientation = self.get_camera_pose(camera_id)
            return camera_pose, camera_orientation
        elif action == 'get_camera_xy_and_yaw':
            camera_id = commond[1]
            x, y, yaw = self.get_camera_xy_and_yaw(camera_id)
            return x, y, yaw
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
        else:
            return commond

    def step(self, commond=None):
        responed = None
        if commond is not None:
            responed = self.exec_action(commond)
        # get_obs
        color_image, depth_image, transformed_point_cloud = self.get_obs()
        if self.args.real_time:
            pose_ts = self.t265_timestep
            rgb_ts = self.rgb_timestep
            if abs(pose_ts - rgb_ts) > self.args.timestamp_threshold_ms:
                print(f"Time synchronization error: Pose {pose_ts}, RGB {rgb_ts}")
                print(abs(pose_ts - rgb_ts))
                return None

        if self.args.vis:
            rgb_save_name = str(self.step_number) + "_rgb.png"
            depth_save_name = str(self.step_number) + "_depth.npy"
            point_cloud_save_name = str(self.step_number) + "_point_cloud.npy"
            cv2.imwrite(os.path.join(self.save_image_dir, rgb_save_name), color_image[:, :, ::-1])
            np.save(os.path.join(self.save_image_dir, depth_save_name), depth_image)
            np.save(os.path.join(self.save_image_dir, point_cloud_save_name), transformed_point_cloud)

        self.step_number += 1
        return responed, color_image, depth_image, transformed_point_cloud

    def get_obs(self):
        # frames = self.rgb_pipeline.wait_for_frames()
        # frames = self.rgb_align_to_color.process(frames)
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        # depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        # 获取位姿
        self.get_robot_pose()
        color_image, depth_image, rgb_timestep = self.rgbd_sensor.capture_rgbd_image()
        point_cloud, rgb_timestep = self.rgbd_sensor.capture_color_point_cloud()
        self.rgb_timestep = rgb_timestep / 1000
        camera_pose, camera_orientation = self.get_camera_pose("FemtoBolt_down")

        # 点云变换
        translation_vector = camera_pose
        rotation_matrix = R.from_quat(quat_wxyz_to_xyzw(camera_orientation)).as_matrix()

        transformed_point_cloud = (rotation_matrix @ point_cloud[:, :3].T).T + translation_vector
        # transformed_point_cloud = np.concatenate((transformed_point_cloud, color_image.reshape([-1, 3])), axis=1)
        transformed_point_cloud = np.concatenate((transformed_point_cloud, point_cloud[:, 3:][:, ::-1]), axis=1)
        return color_image, depth_image, transformed_point_cloud

    def get_tracking_pose(self):
        try:
            frames = self.t265_pipeline.wait_for_frames()
            pose_frame = frames.get_pose_frame()
            timestep = frames.get_timestamp()

            if not pose_frame:
                return None

            pose_data = pose_frame.get_pose_data()

            # Extract position and orientation from the pose data
            position = np.array([pose_data.translation.x,
                                 pose_data.translation.y,
                                 pose_data.translation.z])
            orientation = np.array([pose_data.rotation.w,
                                    pose_data.rotation.x,
                                    pose_data.rotation.y,
                                    pose_data.rotation.z])
            self.t265_timestep = timestep
            return position, orientation, timestep
        except Exception as e:
            print(f"Error getting pose: {e}")
            return None

    def get_robot_pose(self):
        if self.tracking_method == 'odom':
            position, orientation = get_odom_pose()
        elif self.tracking_method == 'T265':
            tracking_position, tracking_orientation, time_step = self.get_tracking_pose()
            R_camera_to_robot = self.camera_list[self.tracking_method]['R_camera_to_robot']
            T_camera_to_robot = self.camera_list[self.tracking_method]['translation_camera_to_robot']
            R_robot_to_camera = R.from_matrix(R_camera_to_robot).inv().as_matrix()
            T_robot_to_camera = R_robot_to_camera @ (-T_camera_to_robot)
            orientation = R.from_matrix(R_camera_to_robot) * R.from_quat(quat_wxyz_to_xyzw(tracking_orientation)) * R.from_matrix(R_robot_to_camera)
            orientation = quat_xyzw_to_wxyz(orientation.as_quat())
            # orientation = orientation.as_euler('xyz')
            position = T_camera_to_robot + R_camera_to_robot @ (
                        tracking_position + R.from_quat(quat_wxyz_to_xyzw(tracking_orientation)).as_matrix() @ T_robot_to_camera)
        else:
            position = np.zeros(3)
            orientation = np.zeros([3, 3])
        self.robot_position = position
        self.robot_orientation = orientation
        # return position, orientation

    def get_robot_xy_and_yaw(self):
        # position, orientation = self.get_robot_pose()
        position = self.robot_position
        orientation = self.robot_orientation
        position_x, position_y = position[0], position[1]
        roll, pitch, yaw = R.from_quat(quat_wxyz_to_xyzw(orientation)).as_euler('xyz')
        return position_x, position_y, yaw

    def get_camera_pose(self, camera_id):
        # position, orientation = self.get_robot_pose()
        position = self.robot_position
        orientation = self.robot_orientation
        camera_pose = position + R.from_quat(quat_wxyz_to_xyzw(orientation)).as_matrix() @ self.camera_list[camera_id][
            'translation_camera_to_robot']
        camera_orientation = R.from_quat(quat_wxyz_to_xyzw(orientation)) * R.from_matrix(
            self.camera_list[camera_id]['R_camera_to_robot'])
        camera_orientation = quat_xyzw_to_wxyz(camera_orientation.as_quat())
        return camera_pose, camera_orientation

    def get_camera_xy_and_yaw(self, camera_id):
        position, orientation = self.get_camera_pose(camera_id)
        position_x, position_y = position[0], position[1]
        roll, pitch, yaw = R.from_quat(quat_wxyz_to_xyzw(self.robot_orientation)).as_euler('xyz')
        return position_x, position_y, yaw


@dataclass
class Config:
    vis: bool = True
    tracking_method: str = "T265"
    save_image_dir: str = "/home/tl/yh/RGBD/images/"
    t265_serial_number: str = "119622110447"
    rgbd_serial_number: str = "CL8M841006W"
    real_time: bool = True
    timestamp_threshold_ms: int = 30


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "/home/tl/yh/output"
    save_image_dir = os.path.join(base_path, current_time, "image")
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    Config.save_image_dir = save_image_dir


    args = Config
    agent = Agent(args)
    # agent('move_forward')

    agent.step(("stop",))
    if args.real_time:
        while True:
            try:
                while True:
                    agent.step(("stop",))
            except KeyboardInterrupt:
                break
            finally:
                agent.t265_pipeline.stop_pipeline()
    else:
        for _ in range(16):
            agent.step(("turn_left",))

    # 合并点云
    point_cloud_list = []
    file_list = os.listdir(agent.save_image_dir)
    for file_one in file_list:
        if "point_cloud" in file_one:
            file_path_one = os.path.join(agent.save_image_dir, file_one)
            point_cloud = np.load(file_path_one)
            point_cloud_list.append(point_cloud)
    point_cloud_list = np.concatenate(point_cloud_list, axis=0)
    scene_point_cloud_dir = os.path.join(base_path, current_time, "scene_point_cloud.txt")
    # np.savetxt(scene_point_cloud_dir, point_cloud_list)
    np.savetxt(scene_point_cloud_dir, point_cloud_list, fmt='%.6f', delimiter=' ', header='X Y Z R G B')
