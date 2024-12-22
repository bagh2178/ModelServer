# ModelServer

ModelServer is a framework for deploying models that isolates the model environment from the project environment and decoups the model code from the project code. Information is transferred between the model code and the project code over tcp. Currently, LLM, VLM and Hexmove Robot are supported.

## Hexmove Robot

**Step 1:** Create a new conda environment ``conda create -n robot``

**Step 2:** Install pyrealsense2<=2.53 from [here](https://github.com/IntelRealSense/librealsense) and install pyorbbecsdk manually from [here](https://github.com/orbbec/pyorbbecsdk)

**Step 3:** Start server ``python start_hexmove.py``

**Step 4:** Deploy Hexmove robot client in your project, code example:

```
from ModelServer.client.hexmove import Hexmove_Client

agent = Hexmove_Client()

# move robot
agent('move_forward')
agent('move_backward')
agent('turn_left')
agent('turn_right')

# get robot pose
position, orientation, timestamp = agent('get_robot_pose')

# get intrinsic
color_intrinsics, depth_intrinsics = agent('get_rgbd_intrinsic', 'camera_id')

# get camera extrinsic
camera_position, camera_orientation, timestamp = agent('get_camera_pose', 'camera_id')

# get RGB-D
rgb_image, depth_image, timestamp = agent('get_rgbd_image', 'camera_id')
rgb_image, depth_image, pose, timestamp = agent('get_rgbd_image', 'camera_id', 'pose')
```

## LLM and VLM

**Step 1:** Create a new conda environment ``conda create -n model_server``

**Step 2:** Install requirements ``pip install -r requirements.txt``

**Step 3 (Optional):** If you want to deploy a VLM, you can install LLaVA according to [here](https://github.com/haotian-liu/LLaVA) 

**Step 4:** Start server

**LLM:** ``conda activate model_server`` and ``python start_llm.py``

**VLM:** ``conda activate model_server`` and ``python start_vlm.py``

**Step 5:** Deploy client in your project

**LLM:** 

```
from ModelServer.client.llm import LLM_Client
llm = LLM_Client()
prompt = 'Hello'
response = llm(prompt)
print(response)
```

**VLM:** 

```
from PIL import Image
from ModelServer.client.vlm import VLM_Client
vlm = VLM_Client()
prompt = 'Describe the objects in the image.'
img_path = '/path/to/the/image'
img = Image.open(img_path)
response = vlm(prompt, img)
print(response)
```
