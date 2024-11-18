# Zoomed In, Diffused Out: Towards Local Degradation-Aware Multi-Diffusion for Extreme Image Super-Resolution

This work and repo builds upon [SeeSR](https://github.com/cswry/SeeSR).

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/cswry/SeeSR.git
cd SeeSR

# create an environment with python >= 3.8
conda create -n seesr python=3.8
conda activate seesr
pip install -r requirements.txt
```

## 🚀 Inference
#### Step 1: Download the pretrained models
- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
- Download the SeeSR and DAPE models from [GoogleDrive](https://drive.google.com/drive/folders/12HXrRGEXUAnmHRaf0bIn-S8XSK4Ku0JO?usp=drive_link) or [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/22042244r_connect_polyu_hk/EiUmSfWRmQFNiTGJWs7rOx0BpZn2xhoKN6tXFmTSGJ4Jfw?e=RdLbvg).

You can put the models into `preset/models`.

#### Step 2: Prepare testing data
You can put the testing images in the `preset/datasets/test_datasets`.

#### Step 3: Running testing command
```
python test_multiseesr_local.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path preset/models/seesr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 \
--upscale 4 \
--local_prompts
```
the tag `--local_prompts` activates the local prompt extraction.

