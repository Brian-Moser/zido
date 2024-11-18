'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint
import RRDBNet_arch as arch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_multiseesr_v2_local import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_rrdbnet():
    model_path = 'preset/models/RRDB_PSNR_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    return model


def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained_orig(args.pretrained_model_path, args.seesr_model_path, subfolder="unet", use_image_cross_attention=True)
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model
    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def main(args, enable_xformers_memory_efficient_attention=True,):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SeeSR")

    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)

    transform_to_tensor = transforms.ToTensor()  # Normalizes between [0, 1]
    transform_to_image = transforms.ToPILImage()  # Converts back to PIL image
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

        
        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")

            image_tensor = transform_to_tensor(validation_image).unsqueeze(0)  # Add batch dimension
            rscale = args.upscale
            tile_size, tile_overlap = (64*8//rscale, 32*8//rscale)
            
            #print(f'rscale: {rscale}, tile_size: {tile_size}, tile_overlap: {tile_overlap}')
            _, _, h, w = image_tensor.size()
            print(f'input size: {h}x{w}')
            
            #h, w = h*rscale // 4, w*rscale // 4
            if args.local_prompts and h*w>tile_size*tile_size:
                tile_size = min(tile_size, min(h, w))
                grid_rows = 0
                cur_x = 0
                while cur_x < w:
                    cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                    grid_rows += 1

                grid_cols = 0
                cur_y = 0
                while cur_y < h:
                    cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                    grid_cols += 1

                validation_prompt = []
                negative_prompt = []
                ram_encoder_hidden_states = []
                for row in range(grid_rows):
                    validation_prompt.append([])
                    negative_prompt.append([])
                    ram_encoder_hidden_states.append([])
                    for col in range(grid_cols):
                        #print(f'row: {row}, col: {col}')
                        if col < grid_cols-1 or row < grid_rows-1:
                            # extract tile from input image
                            ofs_x = max(row * tile_size-tile_overlap * row, 0)
                            ofs_y = max(col * tile_size-tile_overlap * col, 0)
                            # input tile area on total image
                        if row == grid_rows-1:
                            ofs_x = w - tile_size
                        if col == grid_cols-1:
                            ofs_y = h - tile_size

                        input_start_x = ofs_x
                        input_end_x = ofs_x + tile_size
                        input_start_y = ofs_y
                        input_end_y = ofs_y + tile_size
                        crop = image_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                        #print(f'crop size: {crop.size()}')
                        crop_validation_prompt, crop_ram_encoder_hidden_states = get_validation_prompt(args, transform_to_image(crop.squeeze(0)), model)
                        crop_validation_prompt += args.added_prompt
                        crop_negative_prompt = args.negative_prompt
                        validation_prompt[row].append([crop_validation_prompt])
                        negative_prompt[row].append([crop_negative_prompt])
                        ram_encoder_hidden_states[row].append(crop_ram_encoder_hidden_states.detach().cpu())

                validation_image = transform_to_image(image_tensor.squeeze(0))  # Remove batch dimension and convert back to image
            else:
                #print("?")
                validation_image = transform_to_image(image_tensor.squeeze(0))  # Remove batch dimension and convert back to image
                validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)
                validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
                negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            
            if args.save_prompts:
                txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
                if args.local_prompts:
                    file = open(txt_save_path, "w")
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            file.write(str(validation_prompt[row][col]))
                    file.close()
                else:
                    file = open(txt_save_path, "w")
                    file.write(str(validation_prompt))
                    file.close()
            print(f'{validation_prompt}')

            if args.prompts_only:
                continue

            ori_width, ori_height = validation_image.size
            resize_flag = False
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}')

            for sample_idx in range(args.sample_times):
                os.makedirs(f'{args.output_dir}/', exist_ok=True)

            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    image = pipeline(
                            validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                            guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                            start_point=args.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states,
                            latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                            args=args,
                        ).images[0]
                
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale))
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                
                image.save(f'{args.output_dir}/{name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--local_prompts", action='store_true')
    parser.add_argument("--spot_sr", action='store_true')
    parser.add_argument("--prompts_only", action='store_true')
    args = parser.parse_args()
    main(args)



