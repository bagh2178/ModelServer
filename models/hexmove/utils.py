import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_xyzw(quat):
    quat = (quat[1], quat[2], quat[3], quat[0])
    return quat


def quat_xyzw_to_wxyz(quat):
    quat = (quat[3], quat[0], quat[1], quat[2])
    return quat


def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def pose_to_matrix(position, orientation):
    matrix = np.eye(4)
    if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
        matrix[:3, :3] = orientation
    elif isinstance(orientation, np.ndarray) and orientation.shape == (4,):
        matrix[:3, :3] = R.from_quat(quat_wxyz_to_xyzw(orientation)).as_matrix()
    matrix[:3, 3] = position
    return matrix


def matrix_to_pose(matrix):
    orientation = matrix[:3, :3]
    position = matrix[:3, 3]
    orientation = R.from_matrix(orientation).as_quat()
    orientation = quat_wxyz_to_xyzw(orientation)
    return position, orientation


def timestamp_match(timestamp_list, timestamp):
    min_diff = float('inf')
    closest_index = -1
    for i, ts in enumerate(timestamp_list):
        diff = abs(ts - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    return closest_index