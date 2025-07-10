import time
import numpy as np
from ModelServer.client.hexmove import Hexmove_Client
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_id', type=str, help='An string for the dataset id')
parser.add_argument('--episode_index', type=int, help='An integer for the episode index')
args = parser.parse_args()

# 使用从命令行获取的episode_index
dataset_id = args.dataset_id
episode_index = args.episode_index
index = 0
agent = Hexmove_Client()
while True:
    # agent('get_rgb_image_rdt', '336L_arm_right', episode_index, index, 'right_wrist', 'save', "idp3_0414")
    # agent('get_rgb_image_rdt', '336L_arm_left', episode_index, index, 'left_wrist', 'save', "idp3_0414")
    # agent('get_rgb_image_rdt', 'FemtoBolt_down', episode_index, index, 'ext', 'save', "idp3_0414")
    # print(1)
    agent('get_rgbd_image_fast', 'FemtoBolt_up', episode_index, index, 'ext', 'save', dataset_id)
    # agent('get_rgb_image_rdt', '336L_head', episode_index, index, 'ext', 'save', "idp3_0414")
    # agent('get_rgbd_image_rdt', '336L_head', episode_index, index, 'ext', 'save', "idp3_0414")
    # print(2)
    arm_pose = agent('get_arm_pose_idp3', 'arm_right', episode_index, index, 'save', dataset_id)
    print(arm_pose)
    time.sleep(1/18)
    index += 1
    # print(index)
