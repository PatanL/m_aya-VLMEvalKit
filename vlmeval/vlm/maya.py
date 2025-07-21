# vlmeval/vlm/maya.py

import torch
from PIL import Image
import warnings
import sys
import os.path as osp

from .base import BaseModel
from ..smp import *

# This is the main class for your Multimodal Aya model
class MultimodalAya(BaseModel):
    # This tells VLMEvalKit that this model requires a custom codebase.
    INSTALL_REQ = True
    # LLaVA-like models typically handle an image and text as separate inputs, not interleaved.
    INTERLEAVE = False

    def __init__(self, 
                 model_path='maya-multimodal/maya', 
                 model_base='CohereForAI/aya-23-8B',
                 mode='finetuned',
                 projector_path=None,
                 conv_mode='aya',
                 root=None, 
                 **kwargs):
        """
        Args:
            model_path (str): The path to the multimodal model checkpoint (e.g., 'maya-multimodal/maya').
            model_base (str): The path to the base language model (e.g., 'CohereForAI/aya-23-8B').
            mode (str): 'finetuned' or 'pretrained'.
            projector_path (str): Path to the projector weights if mode is 'pretrained'.
            conv_mode (str): The conversation template mode. 'aya' is the default.
            root (str): The absolute path to the cloned repository of your 'llava' codebase.
            **kwargs: Additional generation arguments.
        """
        
        # This is crucial for VLMEvalKit to find your custom model code.
        assert root is not None and osp.isdir(root), \
            "Please provide the absolute path to your 'llava' codebase repository via the 'root' argument."
        sys.path.append(root)

        # We import the necessary functions from your provided codebase.
        try:
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.eval.maya.eval_utils import load_maya_model
        except ImportError as e:
            raise ImportError(f"Failed to import from your custom 'llava' codebase. Please check the path and dependencies. Error: {e}")

        # Store constants and functions as instance variables
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.separator_style = SeparatorStyle.TWO 
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images

        # Load the model using the logic from your eval scripts
        self.tokenizer, self.model, self.image_processor, self.context_len = load_maya_model(
            model_base=model_base,
            model_path=model_path,
            mode=mode,
            projector_path=projector_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto', # This will split the model across your 2 T4s
            trust_remote_code=True
        ).eval()

        self.conv_mode = conv_mode

        # Store generation kwargs
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config.")

    def generate_inner(self, message, dataset=None):
        """
        The core logic for generating a response from the model.
        """
        # Step 1: Separate prompt and image from the VLMEvalKit message format
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.process_images([image], self.image_processor, self.model.config)[0]

        # Step 2: Construct the full prompt using the model's chat template
        # This logic is adapted from your `llava/eval/model_vqa_maya.py`
        if self.model.config.mm_use_im_start_end:
            qs = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            qs = self.DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        # Step 3: Tokenize the prompt
        input_ids = self.tokenizer_image_token(full_prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # Step 4: Run model inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                use_cache=True,
                **self.kwargs
            )

        # Step 5: Decode and clean the output
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # The response might include the prompt, so we need to clean it
        # This part may need adjustment depending on the exact output format
        sep = conv.sep if conv.sep_style != self.separator_style else conv.sep2
        if sep:
            parts = response.split(sep)
            if len(parts) > 1:
                 response = parts[-1].strip()

        return response