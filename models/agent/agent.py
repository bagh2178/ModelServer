import os
import time
from PIL import Image
from .realsense import capture_rgbd_image

class Agent():
    def __init__(self) -> None:
        pass
    
    def __call__(self, x):
        commond = x[0]
        if commond == 'get_rgbd_frame_top':
            serial_number = '337322070914'
            rgb_image, depth_image = capture_rgbd_image(serial_number)
            return rgb_image, depth_image
        elif commond == 'get_rgbd_frame_down':
            serial_number = '327122078142'
            rgb_image, depth_image = capture_rgbd_image(serial_number)
            return rgb_image, depth_image
        elif commond == 'move_forward':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1')
            return 'move_forward done'
        elif commond == 'turn_left':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}" -1')
            return 'turn_left done'
        elif commond == 'turn_right':
            os.system('ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -1.0}}" -1')
            return 'turn_right done'
        elif commond == 'stop':
            return 'success'
        else:
            return x
