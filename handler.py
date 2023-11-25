# this is the huggingface handler file
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available
from typing import Any
import torch
import einops
import torchvision

import numpy as np

class EndpointHandler():
    def __init__(self, model_path: str = "models/StableDiffusion/", inference_config_path: str = "configs/inference/inference-v3.yaml", motion_module: str = "models/Motion_Module/mm_sd_v15.ckpt"):
        
        inference_config = OmegaConf.load(inference_config_path)
        ### >>> create validation pipeline >>> ###
        tokenizer    = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        vae          = AutoencoderKL.from_pretrained(model_path, subfolder="vae")            
        unet         = UNet3DConditionModel.from_pretrained_2d(model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

        if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
        else: assert False

        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs.DDIMScheduler)),
        ).to("cuda")

        self.pipeline = load_weights(
            self.pipeline,
            # motion module
            motion_module_path         = motion_module,
            motion_module_lora_configs = [],
            # image layers
            dreambooth_model_path      = "",
            lora_model_path            = "",
            lora_alpha                 = 0.8,
        ).to("cuda")

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        """
        
        

    def preprocess(self, data):
        """
        preprocess will be called once for each request.
        """
    
    def __call__(self, prompt, negative_prompt, steps, guidance_scale):
        """
        __call__ method will be called once per request. This can be used to
        run inference.
        """
        vids = self.pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=steps, 
            guidance_scale=guidance_scale,
            width= 256,
            height= 256,
            video_length= 5,
            ).videos
        
        videos = einops.rearrange(vids, "b c t h w -> t b c h w")
        n_rows=6
        fps=1
        loop = True
        rescale=False
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(x)
            
        # imageio.mimsave(path, outputs, fps=fps)
        
        # return a gif file as bytes
        return outputs