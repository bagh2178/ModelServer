import os
import glob
import time
import tqdm
import pickle
import numpy as np
from ModelServer import Hexmove_Client


episode_data_dir = '/home/tl/yh/data/idp3_0502/episode_000000'
pose_dir = os.path.join(episode_data_dir, 'pose')
pose_path_list = sorted(glob.glob(os.path.join(pose_dir, '*.pkl')))
pose_list = []
for pose_path in pose_path_list:
    with open(pose_path, 'rb') as f:
        pose = pickle.load(f)
    pose_list.append(pose)


pose_list = pose_list[1:]

agent = Hexmove_Client()

agent('arm_joint_ctrl', 'arm_left', [-0.08390778,0.98915233,-0.40198978,-0.11549967,0.09740978,-0.07945944,0])
agent('arm_joint_ctrl', 'arm_right', [0.18159667,1.03705478,-0.39946033,0.05257756,-0.06639356,0.11031867, 0])

timestamp = pose_list[0]['timestamp']

for i, pose in enumerate(tqdm.tqdm(pose_list)):
    left_arm_joint_ctrl = np.array(list(pose['arm_joint'][:6]) + [pose['arm_gripper_pose'][0]])
    right_arm_joint_ctrl = np.array(list(pose['arm_joint'][6:]) + [pose['arm_gripper_pose'][1]])
    delay = pose['timestamp'] - timestamp
    timestamp = pose['timestamp']
    agent('arm_joint_ctrl', 'arm_left', left_arm_joint_ctrl)
    agent('arm_joint_ctrl', 'arm_right', right_arm_joint_ctrl)