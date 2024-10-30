# ModelServer

ModelServer is a framework for deploying models that isolates the model environment from the project environment and decoups the model code from the project code. Information is transferred between the model code and the project code over tcp. Currently, LLM and VLM are supported.

## Installation

**Step 1:** Create a new conda environment ``conda create -n model_server``

**Step 2:** Install requirements ``pip install -r requirements.txt``

**Step 3(Optional):** If you want to deploy a VLM, you can install LLaVA according to [here](https://github.com/haotian-liu/LLaVA) 

## Start server

**LLM:** ``conda activate model_server`` and ``python start_llm.py``

**VLM:** ``conda activate model_server`` and ``python start_vlm.py``

## Deploy client in your project

**LLM:** 

```
from model_server.llm_client import LLM_Client
llm = LLM_Client()
prompt = 'Hello'
response = llm(prompt)
print(response)
```

**VLM:** 

```
from PIL import Image
from model_server.vlm_client import VLM_Client
vlm = VLM_Client()
prompt = 'Describe the objects in the image.'
img_path = '/path/to/the/image'
img = Image.open(img_path)
response = vlm(prompt, img)
print(response)
```
