import os
import sys
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from model_server.vlm_server import VLM_Server


vlm_server = VLM_Server()
