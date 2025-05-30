import time
import numpy as np
from ModelServer.client.hexmove import Hexmove_Client
import argparse
import sys
import tty
import termios

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_id', type=str, help='An string for the dataset id')
parser.add_argument('--episode_index', type=int, help='An integer for the episode index')
args = parser.parse_args()

# 定义获取单个按键的函数 (仅Linux/Unix)
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# 使用从命令行获取的episode_index
dataset_id = args.dataset_id
episode_index = args.episode_index
index = 0
agent = Hexmove_Client()

print("按 'c' 键进行数据采集, 按 'q' 键退出程序")
print(f"当前 episode: {episode_index}, 起始 index: {index}")

try:
    while True:
        # 获取单个按键输入，不需要回车
        key = getch()
        
        if key == 'q':
            print("\n退出数据采集程序.")
            break
        
        if key == 'c':
            print(f"\n正在采集数据, episode: {episode_index}, index: {index}")
            
            # agent('get_rgb_image_rdt', '336L_arm_right', episode_index, index, 'right_wrist', 'save', "idp3_0414")
            # agent('get_rgb_image_rdt', '336L_arm_left', episode_index, index, 'left_wrist', 'save', "idp3_0414")
            # agent('get_rgb_image_rdt', 'FemtoBolt_down', episode_index, index, 'ext', 'save', "idp3_0414")
            agent('get_rgbd_image_rdt', 'FemtoBolt_down', episode_index, index, 'ext', 'save', dataset_id)
            # agent('get_rgb_image_rdt', '336L_head', episode_index, index, 'ext', 'save', "idp3_0414")
            # agent('get_rgbd_image_rdt', '336L_head', episode_index, index, 'ext', 'save', "idp3_0414")

            # arm_pose = agent('get_arm_pose_idp3', 'arm_right', episode_index, index, 'save', dataset_id)
            # print(arm_pose)
            # time.sleep(2)
            
            index += 1
            print(f"数据采集完成. 下一个 index 将是: {index}")
            print("按 'c' 继续采集, 按 'q' 退出程序")
        if key == 'r':
            print("reset zero pose")
            agent("robot_pose_reset", )
        # 短暂延时，避免CPU占用过高
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n程序被用户中断")
finally:
    print("数据采集程序结束")