# ModelServer

ModelServer is a framework for deploying models on a host computer and remotely calling API over the Internet to read and control robots. Hexmove Robot is currently supported.

## On Host Computer

#### Install

Install ModelServer, in your project environment.

```
pip install git+https://github.com/bagh2178/ModelServer.git
```

or

```
git clone https://github.com/bagh2178/ModelServer.git
pip install -e ModelServer/
```

#### Run

Code example:

```
from ModelServer import Hexmove_Client

agent = Hexmove_Client()

# move robot
agent('move_forward')
agent('move_backward')
agent('turn_left')
agent('turn_right')

# get RGB-D
rgb_image, depth_image, timestamp = agent('get_rgbd_image', 'FemtoBolt_down')
rgb_image, depth_image, timestamp = agent('get_rgbd_image', 'FemtoBolt_down', 'jpg')
rgb_image, depth_image, pose, timestamp = agent('get_rgbd_image', 'FemtoBolt_down', 'pose')
rgb_image, timestamp = agent('get_rgbd_image', 'FemtoBolt_down', 'without_depth')
rgb_image, pose, timestamp = agent('get_rgbd_image', 'FemtoBolt_down', 'pose', 'without_depth')

# get robot pose
pose, timestamp = agent('get_robot_pose')

# get camera intrinsic
color_intrinsics, depth_intrinsics = agent('get_rgbd_intrinsic', 'FemtoBolt_down')

# get camera extrinsic
camera_position, camera_orientation, timestamp = agent('get_camera_pose', 'FemtoBolt_down')
```

You can print to get the usage of the API call.

```
from ModelServer import Hexmove_Client

agent = Hexmove_Client()
print(help(Hexmove_Client))
```

Now, we support a variety of commond in API, including:

```
'get_rgbd_image', <camera_id>, <format>='png', <save>='save', <pose>='pose', <without_depth>='without_depth'
'get_pointcloud', <camera_id>
'get_camera_intrinsic', <camera_id>
'get_camera_extrinsic', <camera_id>
'get_camera_xy_and_yaw', <camera_id>

'robot_pose_reset'
'get_robot_pose'

'robot_move', <goal>
'robot_move_openloop', <goal>

'get_arm_pose', <arm_id>

'arm_enable', <arm_id>
'arm_disable', <arm_id>
'arm_reset', <arm_id>
'arm_prepare', <arm_id>
'arm_open_gripper', <arm_id>
'arm_close_gripper', <arm_id>
'arm_move_camera', <camera_id>, <arm_id>, <position>, <orientation>=None
'arm_move_robot', <arm_id>, <position>, <orientation>=None
'arm_move_local', <arm_id>, <position>, <orientation>=None
'arm_end_pose_ctrl', <arm_id>, <arm_end_pose>
'arm_joint_ctrl', <arm_id>, <arm_joint>
```

## On Hexmove Robot

#### Install

Create environment and install ModelServer.

```
conda create -n modelserver
conda activate modelserver
git clone https://github.com/bagh2178/ModelServer.git
cd ModerServer
pip install -r requirements.txt
```

Install pyrealsense2<=2.53 (T265 camera does not support version>2.53) from [here](https://github.com/IntelRealSense/librealsense) and install pyorbbecsdk manually from [here](https://github.com/orbbec/pyorbbecsdk)

#### Run

Start ModelServer. When the green "READY" appears, ModelServer has started and is waiting for the API to be called.

```
python start_hexmove.py  # run mode: If an error occurs, the server will not exit and display less detailed error
```

or

```
python start_hexmove.py --debug  # debug mode: If an error occurs, the server will exit and display detailed error
```
