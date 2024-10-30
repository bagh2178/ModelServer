import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from llava.conversation import SeparatorStyle, conv_templates
# from llava.model.utils import KeywordsStoppingCriteria
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class LLaVA():  # added by yinhang
    def __init__(self, model_path='liuhaotian/llava-v1.5-7b', model_base=None, device='cuda', conv_mode=None, temperature=0.2, max_new_tokens=512):
        self.model_path = model_path
        self.model_base = model_base
        self.device = device
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.model_base, model_name, device=self.device)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, self.conv_mode, self.conv_mode))
        else:
            self.conv_mode = conv_mode

        self.conv = conv_templates[self.conv_mode].copy()
        if "mpt" in model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = self.conv.roles
        return

    def reset(self):
        self.conv.messages = []

    def load_image(image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def __call__(self, data):
        # image = self.load_image(image_file)
        # Similar operation in model_worker.py
        self.reset()
        query, image = data
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
            else:
                query = DEFAULT_IMAGE_TOKEN + '\n' + query
            self.conv.append_message(self.conv.roles[0], query)
            image = None
        else:
            # later messages
            self.conv.append_message(self.conv.roles[0], query)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return (outputs,)