# host a server that can be accessed by any post request
# and return the result of the model

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
# this is the huggingface handler file

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download, try_to_load_from_cache

import os
import json
import base64

from safetensors import safe_open

from diffusers.utils.import_utils import is_xformers_available
from typing import Any
import torch
import imageio
import torchvision
import random
import numpy as np
from einops import rearrange

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora


current_model = "backup"

class EndpointHandler():
    def __init__(self, model_path: str = "bluestarburst/AnimateDiff-SceneFusion"):

        # inference_config_path = "configs/inference/inference-v3.yaml"
        inference_config_path = hf_hub_download(repo_id="bluestarburst/AnimateDiff-SceneFusion", filename="configs/inference/inference-v3.yaml")
        print(inference_config_path)

        inference_config = OmegaConf.load(inference_config_path)

        # inference_config = {'unet_additional_kwargs': {'unet_use_cross_frame_attention': False, 'unet_use_temporal_attention': False, 'use_motion_module': True, 'motion_module_resolutions': [1, 2, 4, 8], 'motion_module_mid_block': False, 'motion_module_decoder_only': False, 'motion_module_type': 'Vanilla', 'motion_module_kwargs': {'num_attention_heads': 8, 'num_transformer_block': 1, 'attention_block_types': ['Temporal_Self', 'Temporal_Self'], 'temporal_position_encoding': True, 'temporal_position_encoding_max_len': 24, 'temporal_attention_dim_div': 1}}, 'noise_scheduler_kwargs': {'DDIMScheduler': {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'linear', 'steps_offset': 1, 'clip_sample': False}, 'EulerAncestralDiscreteScheduler': {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'linear'}, 'KDPM2AncestralDiscreteScheduler': {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'linear'}}}

        ### >>> create validation pipeline >>> ###
        tokenizer    = CLIPTokenizer.from_pretrained(model_path, subfolder="models/StableDiffusion/tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="models/StableDiffusion/text_encoder")
        vae          = AutoencoderKL.from_pretrained(model_path, subfolder="models/StableDiffusion/vae").to("cuda")

        unet_model_path = hf_hub_download(repo_id="bluestarburst/AnimateDiff-SceneFusion", filename="models/StableDiffusion/unet/diffusion_pytorch_model.bin")
        unet_config_path = hf_hub_download(repo_id="bluestarburst/AnimateDiff-SceneFusion", filename="models/StableDiffusion/unet/config.json")

        print(unet_model_path)

        unet         = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path=unet_model_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs), config_path=unet_config_path)

        self.latents = []
        # inv_latent_path = f"{OUTPUT_DIR}/inv_latents/ddim_latent-1.pt"
        for i in range(1, 20):
            inv_latent_path = hf_hub_download(repo_id="bluestarburst/AnimateDiff-SceneFusion", filename=f"models/Motion_Module/{current_model}/inv_latents/ddim_latent-{i}.pt")
            self.latents.append(torch.load(inv_latent_path).to(torch.float))
            print(self.latents[i-1].shape, self.latents[i-1].dtype)

        # torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
        else: assert False

        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs.DDIMScheduler))
        ).to("cuda")

        # huggingface download motion module from bluestarburst/AnimateDiff-SceneFusion/models/Motion_Module/mm_sd_v15.ckpt

        # motion_module = hf_hub_download(repo_id="bluestarburst/AnimateDiff-SceneFusion", filename="models/Motion_Module/mm_sd_v15.ckpt")
        motion_module = hf_hub_download(repo_id="bluestarburst/AnimateDiff-SceneFusion", filename=f"models/Motion_Module/{current_model}/mm.pth")
        # LORA_DREAMBOOTH_PATH="models/DreamBooth_LoRA/toonyou_beta3.safetensors"

        # LORA_DREAMBOOTH_PATH = ""
        LORA_DREAMBOOTH_PATH = hf_hub_download(repo_id="bluestarburst/AnimateDiff-SceneFusion", filename="models/DreamBooth_LoRA/toonyou_beta3.safetensors")

        # self.pipeline = load_weights(
        #     self.pipeline,
        #     # motion module
        #     motion_module_path         = motion_module,
        #     motion_module_lora_configs = [],
        #     # image layers
        #     dreambooth_model_path      = "",
        #     lora_model_path            = "",
        #     lora_alpha                 = 0.8,
        # ).to("cuda")

        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        missing, unexpected = self.pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0


        # FIX THIS
        if LORA_DREAMBOOTH_PATH != "":
            if LORA_DREAMBOOTH_PATH.endswith(".ckpt"):
                state_dict = torch.load(LORA_DREAMBOOTH_PATH)
                self.pipeline.unet.load_state_dict(state_dict)

            elif LORA_DREAMBOOTH_PATH.endswith(".safetensors"):
                state_dict = {}
                with safe_open(LORA_DREAMBOOTH_PATH, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

                is_lora = all("lora" in k for k in state_dict.keys())
                if not is_lora:
                    base_state_dict = state_dict
                else:
                    base_state_dict = {}
                    with safe_open("", framework="pt", device="cpu") as f:
                        for key in f.keys():
                            base_state_dict[key] = f.get_tensor(key)

                # vae
                converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, self.pipeline.vae.config)
                self.pipeline.vae.load_state_dict(converted_vae_checkpoint)
                # unet
                converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, self.pipeline.unet.config)
                self.pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                # text_model (TODO: problem here)
                # converted_test_encoder_checkpoint = convert_ldm_clip_checkpoint(base_state_dict)
                # pipeline.text_encoder = converted_test_encoder_checkpoint

                # import pdb
                # pdb.set_trace()
                if is_lora:
                    self.pipeline = convert_lora(self.pipeline, state_dict)
                    # self.pipeline = convert_lora(self.pipeline, state_dict, alpha=model_config.lora_alpha)

        self.pipeline.to("cuda")

    def __call__(self, data : Any):
        """
        __call__ method will be called once per request. This can be used to
        run inference.
        """
        
        print(data)

        prompt = data.pop("prompt", "")
        prompt = f"camera panning right to left, {prompt}, masterpiece, best quality"
        negative_prompt = data.pop("negative_prompt", "")
        negative_prompt += ",easynegative,bad_construction,bad_structure,bad_wail,bad_windows,blurry,cloned_window,cropped,deformed,disfigured,error,extra_windows,extra_chimney,extra_door,extra_structure,extra_frame,fewer_digits,fused_structure,gross_proportions,jpeg_artifacts,long_roof,low_quality,structure_limbs,missing_windows,missing_doors,missing_roofs,mutated_structure,mutation,normal_quality,out_of_frame,owres,poorly_drawn_structure,poorly_drawn_house,signature,text,too_many_windows,ugly,username,uta,watermark,worst_quality"
        steps = data.pop("steps", 25)
        guidance_scale = data.pop("guidance_scale", 12.5)
        
        
        print("data: " + str(prompt) + str(negative_prompt) + str(steps) + str(guidance_scale))

        # print(f"current seed: {torch.initial_seed()}")
        
        # random seed
        # torch.manual_seed(0)
        
        # get random latent from self.latents
        latent = self.latents[random.randint(0, len(self.latents)-1)]
        
        print(f"sampling {prompt} ...")
        vids = self.pipeline(
            prompt,
            negative_prompt     = negative_prompt,
            num_inference_steps = steps,
            guidance_scale      = guidance_scale,
            width               = 256,
            height              = 256,
            video_length        = 5,
            latents             = latent,
        ).videos

        # vids = self.pipeline(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     num_inference_steps=steps,
        #     guidance_scale=guidance_scale,
        #     width= 256,
        #     height= 256,
        #     video_length= 5,
        #     ).videos

        videos = rearrange(vids, "b c t h w -> t b c h w")
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

        path = "output.gif"
        imageio.mimsave(path, outputs, fps=fps, loop=0)

        # open the file as binary and read the data
        with open(path, mode="rb") as file:
            file_content = file.read()
        # return json response with binary data
        # Encode the binary data using Base64
        base64_encoded_content = base64.b64encode(file_content).decode("utf-8")

        # Create a JSON object with the Base64-encoded content
        json_data = {
            "filename": "output.gif",
            "content": base64_encoded_content
        }

        # Convert the JSON object to a JSON-formatted string
        return json.dumps(json_data)


# This is the entry point for the serverless function.
# This function will be called during inference time.

# create an instance of the handler
handler = EndpointHandler()

# create a flask app instance and have static_url_path point to docs/src
app = Flask(__name__, static_url_path='/docs')

#allow any origin to make a request
CORS(app)

# define a route which will be called on inference
@app.route('/scene', methods=['POST'])
def inference():
    print("inference called")
    # get the request data
    data = request.get_json(force=True)
    
    real_data = data["inputs"]
    
    print(real_data)
    
    # call the handler
    result = handler(real_data)
    # return the result back
    return result

# GET request to check if the server is running
@app.route('/')
def index():
    # send html file as response at docs/index.html
    return app.send_static_file('index.html')

# run the app
if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")