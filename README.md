# ModelServer

ModelServer is a framework for deploying models that isolates the model environment from the project environment and decoups the model code from the project code. Information is transferred between the model code and the project code over tcp. Currently, Hexmove Robot are supported. You can install ModelServer on robot and upper computer, then call api on upper computer to read and control robot.

## Hexmove Robot

**Step 1:** Environment

Create environment and install ModelServer.

```
conda create -n modelserver
conda activate modelserver
git clone https://github.com/bagh2178/ModelServer.git
cd ModerServer
pip install -r requirements.txt
```

Install pyrealsense2<=2.53 from [here](https://github.com/IntelRealSense/librealsense) and install pyorbbecsdk manually from [here](https://github.com/orbbec/pyorbbecsdk)

**Step 2:** Start

Start ModelServer. When the green "READY" appears, ModelServer has started and is waiting for the API to be called.

```
python start_hexmove.py
```



## Upper Computer

**Step 1:** Environment

Install ModelServer, in your project environment.

```
conda activate <your_project_env>
git clone https://github.com/bagh2178/ModelServer.git
pip install -e ModelServer/
```

**Step 2:** API

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
get_robot_pose
get_rgbd_intrinsic
```
