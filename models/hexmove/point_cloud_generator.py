import cv2
import numpy as np
import open3d as o3d

def generate_point_cloud(depth_image, rgb_image, fx, fy, cx, cy, depth_scale, robot_x, robot_y, robot_yaw, R_camera_to_robot, translation_camera_to_robot):
    # 确保深度图数据类型为float32
    if depth_image.dtype != np.float32:
        depth_image = depth_image.astype(np.float32)

    # 将深度图从毫米转换为米
    depth_image *= depth_scale

    # 创建网格坐标
    u, v = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
    z = depth_image

    # 忽略无效深度值
    valid_mask = (z > 0) & (z <= 15)
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]

    # 计算XYZ坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.vstack((x, y, z)).T

    # 获取对应的颜色
    colors = rgb_image[v, u] / 255.0  # 归一化颜色值

    # 将点云从相机坐标系转换到机器人坐标系
    points_robot = (R_camera_to_robot @ points.T).T + translation_camera_to_robot

    # 构建从机器人坐标系到世界坐标系的变换矩阵
    R_robot_to_world = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
        [np.sin(robot_yaw), np.cos(robot_yaw), 0],
        [0, 0, 1]
    ])
    translation_robot_to_world = np.array([robot_x, robot_y, 0])

    # 将点云从机器人坐标系转换到世界坐标系
    points_world = (R_robot_to_world @ points_robot.T).T + translation_robot_to_world

    # 创建Open3D点云对象
    point_cloud_world = o3d.geometry.PointCloud()
    point_cloud_world.points = o3d.utility.Vector3dVector(points_world)
    point_cloud_world.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud_world, points_world, colors