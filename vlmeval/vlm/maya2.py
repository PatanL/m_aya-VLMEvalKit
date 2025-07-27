# vlmeval/vlm/maya.py

import torch
from PIL import Image
import warnings
import sys
import os
import os.path as osp
import requests
from io import BytesIO
from typing import Optional, Literal

# --- Imports moved to the top to fix NameError ---
from transformers import AutoTokenizer, AutoConfig, TextStreamer, AutoModelForCausalLM

from .base import BaseModel
from ..smp import *

# ====================================================================================
# The load_maya_model function is now part of this file
# ====================================================================================

def load_maya_model(model_base: str, model_path : str, projector_path : Optional[str] = None, mode = Literal['pretrained','finetuned'], **kwargs):
    """ 
    Function that helps load a trained Maya model.
    This is a modified version to correctly handle quantization arguments.
    """
    # These imports are now scoped within the function as they come from the 'maya' repo
    from llava.model.language_model.llava_cohere import LlavaCohereForCausalLM, LlavaCohereConfig
    from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    device_map = 'auto'
    # Combine kwargs with default loading parameters
    loading_kwargs = {"device_map": device_map, "torch_dtype": torch.float16}
    loading_kwargs.update(kwargs) # This will add load_in_8bit=True if provided

    ## Instantiating tokenizer and model base
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
    cfg_pretrained = LlavaCohereConfig.from_pretrained(model_path) 

    if mode == 'pretrained':
        # The 'pretrained' mode might still cause OOM, finetuned is safer for quantization
        model = LlavaCohereForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **loading_kwargs)

        ## Loading Projector layer weights
        mm_projector_weights = torch.load(projector_path, map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else: # finetuned mode
        model = LlavaCohereForCausalLM.from_pretrained(model_path, config=cfg_pretrained, **loading_kwargs)

    ## Loading image processor
    image_processor = None

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    # This is the memory-intensive step. It is handled more carefully by bitsandbytes
    # when load_in_8bit=True is passed to from_pretrained.
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return model, tokenizer, image_processor, context_len


# ====================================================================================
# Main MultimodalAya Class Definition
# ====================================================================================

class MultimodalAya(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, 
                 model_path='maya-multimodal/maya', 
                 model_base='CohereForAI/aya-23-8B',
                 mode='finetuned',
                 projector_path=None,
                 conv_mode='aya',
                 root=None, 
                 **kwargs):
        
        assert root is not None and osp.isdir(root), \
            "Please provide the absolute path to your 'maya' codebase for its utilities."
        sys.path.append(root)

        try:
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import process_images, tokenizer_image_token
        except ImportError as e:
            raise ImportError(f"Failed to import from your custom 'maya' codebase. Error: {e}")

        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.separator_style = SeparatorStyle.TWO 
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images

        # Load the model using the local, corrected function
        self.model, self.tokenizer, self.image_processor, self.context_len = load_maya_model(
            model_base=model_base,
            model_path=model_path,
            mode=mode,
            projector_path=projector_path,
            **kwargs # Pass kwargs like load_in_8bit=True here
        )

        self.conv_mode = conv_mode
        self.generation_kwargs = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1)
        warnings.warn(f"Using generation kwargs: {self.generation_kwargs}")

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.process_images([image], self.image_processor, self.model.config)[0]

        qs = self.DEFAULT_IMAGE_TOKEN + '\\n' + prompt

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(full_prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                use_cache=True,
                **self.generation_kwargs
            )

        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        sep = conv.sep if conv.sep_style != self.separator_style else conv.sep2
        if sep and sep in response:
            parts = response.split(sep)
            response = parts[-1].strip()
        
        return response