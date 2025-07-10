save_image_dir = '/home/tl/yh/ModelServer/data/hexmove/images/{}'
save_rdt_image_dir = '/home/tl/yh/data/{}/episode_{:0>6}/rgb/{}'
save_rdt_arm_pose_dir = '/home/tl/yh/data/{}/episode_{:0>6}/pose'

import numpy as np
from scipy.spatial.transform import Rotation as R

device_list = {
    'D435i_top': {
        'serial_number': '337322070914',
        'T_camera_to_robot': np.array([[0, 0, 1, 0.35],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 1.3],
                                       [0, 0, 0, 1]]),
    },
    'D435i_down': {
        'serial_number': '327122078142',
        'T_camera_to_robot': np.array([[0, 0, 1, 0.43],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 0.88],
                                       [0, 0, 0, 1]]),
    },
    'T265': {
        'serial_number': '119622110447',
        'T_camera_to_robot': np.array([[0, 0, -1, 0.35],
                                       [-1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1]]),
    },
    'FemtoBolt_up': {
        'serial_number': 'CL8M841005A',
        'T_camera_to_robot': np.array([[0, 0, 1, 0.33],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 1.64],
                                       [0, 0, 0, 1]]),
        'rotation': R.from_euler('xyz', (-np.pi / 180 * 60, 0, 0)).as_matrix()
    },
    'FemtoBolt_down': {
        'serial_number': 'CL8M841006W',
        'T_camera_to_robot': np.array([[0, 0, 1, 0.43],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 0.88],
                                       [0, 0, 0, 1]]),
        'rotation': R.from_euler('xyz', (-np.pi / 180 * 27, 0, 0)).as_matrix()
    },
    '336L_head': {
        'serial_number': 'CP82841000DE',
        'T_camera_to_robot': np.array([[0, 0, 1, 0.43],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 0.88],
                                       [0, 0, 0, 1]]),
    },
    "336L_arm_right": {
        "serial_number": "CP828410001R",
        'T_camera_to_robot': np.array([[0, 0, 1, 0.35],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 1.0],
                                       [0, 0, 0, 1]]),
    },
    "336L_arm_left": {
        "serial_number": "CP84B410000V",
        'T_camera_to_robot': np.array([[0, 0, 1, 0.35],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 1.0],
                                       [0, 0, 0, 1]]),
    },
    'arm_left': {
        'can': 'can0',
        'T_arm_to_robot': np.array([[1, 0, 0, 0.30],
                                       [0, 1, 0, 0.335],
                                       [0, 0, 1, 0.88],
                                       [0, 0, 0, 1]]),
    },
    'arm_right': {
        'can': 'can1',
        'T_arm_to_robot': np.array([[1, 0, 0, 0.30],
                                       [0, 1, 0, -0.335],
                                       [0, 0, 1, 1.11],
                                       [0, 0, 0, 1]]),
    },
}

# 处理rotation合并到T_camera_to_robot
for device_id in device_list:
    if 'rotation' in device_list[device_id]:
        rotation = np.eye(4)
        rotation[:3, :3] = device_list[device_id]['rotation']
        device_list[device_id]['T_camera_to_robot'] = device_list[device_id]['T_camera_to_robot'] @ rotation 