# ModelServer

ModelServer is a framework for deploying models on a host computer and remotely calling API over the Internet to read and control robots. Hexmove Robot is currently supported.

## On Upper Computer

**Step 1: Install**

Install ModelServer, in your project environment.

```
pip install git+https://github.com/bagh2178/ModelServer.git
```

or

```
git clone https://github.com/bagh2178/ModelServer.git
pip install -e ModelServer/
```

**Step 2: Run**

Code example:

```
from ModelServer import Hexmove_Client

agent = Hexmove_Client()

# move robot
agent('move_forward')
agent('move_backward')
agent('turn_left')
agent('turn_right')

# get robot pose
pose, timestamp = agent('get_robot_pose')

# get camera intrinsic
color_intrinsics, depth_intrinsics = agent('get_rgbd_intrinsic', 'camera_id')

# get camera extrinsic
camera_position, camera_orientation, timestamp = agent('get_camera_pose', 'camera_id')

# get RGB-D
rgb_image, depth_image, timestamp = agent('get_rgbd_image', 'camera_id')
rgb_image, depth_image, pose, timestamp = agent('get_rgbd_image', 'camera_id', 'pose')
rgb_image, timestamp = agent('get_rgbd_image', 'camera_id', 'without_depth')
rgb_image, pose, timestamp = agent('get_rgbd_image', 'camera_id', 'pose', 'without_depth')
```

You can print to get the usage of the API call.

```
from ModelServer import Hexmove_Client

agent = Hexmove_Client()
print(help(Hexmove_Client))
```

Now, we support a variety of commond in API, including:

```
robot_pose_reset
get_rgbd_image
get_rgbd_image_rdt
get_pointcloud
get_camera_intrinsic
get_camera_extrinsic
get_camera_xy_and_yaw
get_robot_pose
robot_move
robot_move_openloop
get_arm_pose
get_arm_pose_rdt
arm_reset
arm_prepare
arm_open_gripper
arm_close_gripper
arm_enable
arm_disable
arm_move_camera
arm_move_robot
arm_move_local
arm_end_pose_ctrl
arm_joint_ctrl
```

## On Hexmove Robot

**Step 1: Install**

Create environment and install ModelServer.

```
conda create -n modelserver
conda activate modelserver
git clone https://github.com/bagh2178/ModelServer.git
cd ModerServer
pip install -r requirements.txt
```

Install pyrealsense2<=2.53 from [here](https://github.com/IntelRealSense/librealsense) and install pyorbbecsdk manually from [here](https://github.com/orbbec/pyorbbecsdk)

**Step 2: Run**

Start ModelServer. When the green "READY" appears, ModelServer has started and is waiting for the API to be called.

```
python start_hexmove.py
```
