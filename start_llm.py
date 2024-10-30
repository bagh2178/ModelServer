import os
import sys
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from model_server.llm_server import LLM_Server


llm_server = LLM_Server()
