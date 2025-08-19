#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-05-08
################################################################

import copy
import json
import threading
import time
import multiprocessing
import numpy as np

import rospy
from sensor_msgs.msg import JointState
from xpkg_arm_msgs.msg import XmsgArmJointParam
from xpkg_arm_msgs.msg import XmsgArmJointParamList


class JointTrack:

    def __init__(self):
        ### ros node
        rospy.init_node("joint_track", anonymous=True)
        self.__rate = rospy.Rate(50.0)

        ### publisher
        self.__joint_ctrl_pub = rospy.Publisher(
            '/xtopic_arm/joints_cmd',
            XmsgArmJointParamList,
            queue_size=10,
        )

        ### subscriber
        self.__joint_state_sub = rospy.Subscriber(
            '/xtopic_arm/joint_states',
            JointState,
            self.__joint_state_callback,
        )
        self.__joint_state_sub

        ### variable
        self.__delta_lr_max = 0.10
        self.__delta_dm_max = 0.05
        # target
        self.__target_change_thresh = 0.2
        # deadzone - for each joint
        self.__deadzone_thresh = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

        self.__target_joints = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array(
                [0.5, 0.523598775598, 2.09439265359, 1.57, -1.0472, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array(
                [-0.5, 0.523598775598, 2.09439265359, -1.57, -1.0472, 0.0]),
        ]

        self.__target_joints_idx = 0
        self.__joint_num = self.__target_joints[0].shape[0]
        # state
        self.__pos_lock = threading.Lock()
        self.__cur_pos = np.zeros(self.__joint_num)
        
        # control process variables
        self.__control_process = None
        self.__control_queue = multiprocessing.Queue()
        self.__control_running = multiprocessing.Value('i', 0)
        
        # control
        self.__ctrl_msg = XmsgArmJointParamList()
        for _ in range(self.__joint_num):
            param = XmsgArmJointParam()
            param.mode = "position_mode"
            param.position = 0.0
            param.velocity = 0.0
            param.effort = 0.0
            param.extra_param = json.dumps({"braking_state": False})
            self.__ctrl_msg.joints.append(param)
        
        # start control process
        self.__start_control_process()

        # finish log
        print("joint track node init finished")

    def __joint_state_callback(self, msg: JointState):
        # update state
        with self.__pos_lock:
            self.__cur_pos = np.array(msg.position)
    
    def __start_control_process(self):
        """Start the 50Hz control process"""
        self.__control_running.value = 1
        self.__control_process = multiprocessing.Process(
            target=self.__control_worker,
            args=(self.__control_queue, self.__control_running)
        )
        self.__control_process.start()
    
    def __control_worker(self, control_queue, control_running):
        """50Hz control worker process"""
        # Initialize ROS node in the new process
        rospy.init_node("joint_control_worker", anonymous=True)
        rate = rospy.Rate(50.0)
        
        # Create publisher in the new process
        joint_ctrl_pub = rospy.Publisher(
            '/xtopic_arm/joints_cmd',
            XmsgArmJointParamList,
            queue_size=10,
        )
        
        # Wait for publisher to be ready
        rospy.sleep(0.5)
        
        # Create control message template
        ctrl_msg = XmsgArmJointParamList()
        for _ in range(6):  # Assuming 6 joints
            param = XmsgArmJointParam()
            param.mode = "position_mode"
            param.position = 0.0
            param.velocity = 0.0
            param.effort = 0.0
            param.extra_param = json.dumps({"braking_state": False})
            ctrl_msg.joints.append(param)
        
        current_positions = np.zeros(6)
        
        while control_running.value and not rospy.is_shutdown():
            try:
                # Check for new target positions
                if not control_queue.empty():
                    current_positions = control_queue.get_nowait()
                
                # Update control message
                for i in range(6):
                    ctrl_msg.joints[i].position = current_positions[i]
                
                # Publish control message
                joint_ctrl_pub.publish(ctrl_msg)
                
                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Control worker error: {e}")
                break
    
    def read_joint_angles(self):
        """Read current joint angles
        
        Returns:
            np.array: Current joint positions
        """
        with self.__pos_lock:
            return copy.deepcopy(self.__cur_pos)
    
    def set_joint_angles(self, target_angles, timeout=10.0):
        """Set joint angles and wait for completion
        
        Args:
            target_angles (np.array or list): Target joint angles
            timeout (float): Maximum wait time in seconds
            
        Returns:
            bool: True if target reached within timeout, False otherwise
        """
        if not isinstance(target_angles, np.ndarray):
            target_angles = np.array(target_angles)
        
        if len(target_angles) != self.__joint_num:
            rospy.logerr(f"Invalid joint angles length: expected {self.__joint_num}, got {len(target_angles)}")
            return False
        
        # Send target to control process
        try:
            self.__control_queue.put(target_angles)
        except Exception as e:
            rospy.logerr(f"Failed to send target angles: {e}")
            return False
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_pos = self.read_joint_angles()
            delta = np.abs(target_angles - current_pos)
            
            # Check if all joints are within deadzone
            if np.all(delta < self.__deadzone_thresh):
                return True
            
            time.sleep(0.1)  # Check every 100ms
        
        return False
    
    def stop_control_process(self):
        """Stop the control process"""
        if self.__control_process is not None:
            self.__control_running.value = 0
            self.__control_process.join(timeout=2.0)
            if self.__control_process.is_alive():
                self.__control_process.terminate()
                self.__control_process.join()

    def work(self):
        """Main work loop - cycle through target positions"""
        print("Starting work loop - cycling through 4 target positions")
        
        while not rospy.is_shutdown():
            # Get current target position
            target_pos = self.__target_joints[self.__target_joints_idx]
            print(f"Moving to target position {self.__target_joints_idx}: {target_pos}")
            
            # Use the new set_joint_angles function to move to target
            success = self.set_joint_angles(target_pos, timeout=15.0)
            
            if success:
                print(f"Successfully reached target position {self.__target_joints_idx}")
                # Move to next target
                self.__target_joints_idx = (self.__target_joints_idx + 1) % len(self.__target_joints)
                # Wait a bit before moving to next position
                time.sleep(1.0)
            else:
                print(f"Failed to reach target position {self.__target_joints_idx} within timeout")
                # Still move to next target to avoid getting stuck
                self.__target_joints_idx = (self.__target_joints_idx + 1) % len(self.__target_joints)
                time.sleep(0.5)
    
    def __del__(self):
        """Destructor to clean up control process"""
        self.stop_control_process()


def main():
    joint_track = JointTrack()
    try:
        joint_track.work()
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
    finally:
        joint_track.stop_control_process()
        print("Joint track node stopped")


if __name__ == '__main__':
    main()

