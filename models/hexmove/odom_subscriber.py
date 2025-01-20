import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import time
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(q):
    """
    将四元数转换为欧拉角（Roll, Pitch, Yaw）。
    :param q: 四元数 [w, x, y, z]
    :return: 欧拉角 (roll, pitch, yaw) 以弧度为单位
    """
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(2.0 * (w * y - z * x))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw

class OdomSubscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.first_message_received = False  # 标记是否已经收到过一次消息
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.position_x = None
        self.position_y = None
        self.yaw = None

    def listener_callback(self, msg):
        if not self.first_message_received:  # 如果是第一次接收到消息
            # 提取当前位置
            self.position_x = msg.pose.pose.position.x
            self.position_y = msg.pose.pose.position.y
            self.position_z = msg.pose.pose.position.z

            # 提取当前角度
            self.orientation_w = msg.pose.pose.orientation.w
            self.orientation_x = msg.pose.pose.orientation.x
            self.orientation_y = msg.pose.pose.orientation.y
            self.orientation_z = msg.pose.pose.orientation.z

            # 提取并计算当前偏航角（Yaw）
            orientation_q = [msg.pose.pose.orientation.x,
                             msg.pose.pose.orientation.y,
                             msg.pose.pose.orientation.z,
                             msg.pose.pose.orientation.w]
            # self.roll, self.pitch, self.yaw = quaternion_to_euler(orientation_q)
            self.roll, self.pitch, self.yaw = R.from_quat(orientation_q).as_euler('xyz')

            # 标记已收到消息，并取消订阅
            self.first_message_received = True
            self.destroy_subscription(self.subscription)  # 取消订阅

def get_odom_pose():
    time.sleep(1.5)

    rclpy.init()
    
    odom_subscriber = OdomSubscriber()
    
    try:
        while rclpy.ok() and not odom_subscriber.first_message_received:
            rclpy.spin_once(odom_subscriber, timeout_sec=0.1)
    finally:
        odom_subscriber.destroy_node()
        rclpy.shutdown()

    position = (odom_subscriber.position_x, odom_subscriber.position_y, odom_subscriber.position_z)
    orientation = (odom_subscriber.orientation_x, odom_subscriber.orientation_y, odom_subscriber.orientation_z, odom_subscriber.orientation_w)
    timestamp = 0
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = R.from_quat(orientation).as_matrix()
    robot_pose[:3, 3] = np.array(position)

    return robot_pose, timestamp

def get_odom_xy_and_yaw():
    position, orientation = get_odom_pose()
    position_x, position_y = position[0], position[1]
    orientation = (orientation[1], orientation[2], orientation[3], orientation[0])
    roll, pitch, yaw = R.from_quat(orientation).as_euler('xyz')
    return (position_x, position_y, yaw)

def get_camera_xy_and_yaw(camera_offset=0.3):
    # 假设这里已经通过get_robot_pose函数得到了机器人的位姿信息
    position, orientation = get_odom_pose()
    robot_x, robot_y = position[0], position[1]
    orientation = (orientation[1], orientation[2], orientation[3], orientation[0])
    roll, pitch, yaw = R.from_quat(orientation).as_euler('xyz')
    robot_yaw = yaw

    # 计算相机相对机器人的偏移量
    camera_x_offset = camera_offset * math.cos(robot_yaw)  # 相机与机器人之间的距离
    camera_y_offset = camera_offset * math.sin(robot_yaw)

    # 计算相机的世界坐标
    camera_x = robot_x + camera_x_offset
    camera_y = robot_y + camera_y_offset

    # 相机的yaw角与机器人相同
    camera_yaw = robot_yaw

    return (camera_x, camera_y, camera_yaw)

