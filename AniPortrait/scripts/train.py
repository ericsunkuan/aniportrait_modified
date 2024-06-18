import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils.util import get_fps, read_frames, save_videos_grid

import argparse
import os
import ffmpeg
import random
from datetime import datetime
from pathlib import Path
from typing import List
import subprocess
import av
import numpy as np
import cv2
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

from src.audio_models.model import Audio2MeshModel
from src.audio_models.pose_model import Audio2PoseModel
from src.utils.audio_util import prepare_audio_feature
from src.utils.mp_utils  import LMKExtractor
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.pose_util import project_points, smooth_pose_seq
from src.utils.frame_interpolation import init_frame_interpolation_model, batch_images_interpolation_tool


from scripts.audio2vid_func import a2v

# Assuming you have a Dataset class that returns (ref_image, audio_clip, ref_video)
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import cv2

def read_video_to_rgb(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize an empty list to store the frames
    video = []

    while(cap.isOpened()):
        # Read the video frame by frame
        ret, frame = cap.read()

        if ret:
            # OpenCV reads frames in BGR format, convert it to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Append the frame to the video list
            video.append(rgb_frame)
        else:
            break

    # Release the video capture
    cap.release()

    # Return the list of RGB frames
    return video



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_image_path = ""
    audio_path = ""
    video_path = ""
    

    # default parameters
    config_path = './configs/prompts/animation_audio.yaml'
    width = 512
    height = 512
    L = 300 # generating frame num
    seed = 42
    cfg = 3.5
    steps = 25
    fps = 30
    acc = True
    fi_step = 3

    config = OmegaConf.load(config_path)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
        
    audio_infer_config = OmegaConf.load(config.audio_inference_config)



    # prepare model   
    
    a2m_model = Audio2MeshModel(audio_infer_config['a2m_model'])
    a2m_model.load_state_dict(torch.load(audio_infer_config['pretrained_model']['a2m_ckpt']), strict=False)
    a2m_model.cuda()


    a2p_model = Audio2PoseModel(audio_infer_config['a2p_model'])
    a2p_model.load_state_dict(torch.load(audio_infer_config['pretrained_model']['a2p_ckpt']), strict=False)
    a2p_model.cuda()

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")


    pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(device="cuda", dtype=weight_dtype) # not use cross attention

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(seed)

    # width, height = args.W, args.H



    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)


    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer(forehead_edge=False)

















    # Initialize your model
    # model = YourModel().to(device)

    # Define your loss function
    criterion = nn.MSELoss()

    # Define your optimizer
    optimizer = Adam(a2m_model.parameters(), lr=0.001)


    # Warmup setup
    num_epochs = 30
    num_warmup_steps = int(0.1 * num_epochs)  # 10% of training steps for warmup
    num_training_steps = num_epochs 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    # num_warmup_steps = int(0.1 * num_epochs * len(train_loader))  # 10% of training steps for warmup
    # num_training_steps = num_epochs * len(train_loader)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    label = read_video_to_rgb(video_path)

    # Training loop
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}")
        running_loss = 0.0  # Initialize running loss for the epoch
        # progress = tqdm(enumerate(train_loader), total=len(train_loader))
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # inputs = inputs.to(device)
        labels = labels.to(device)
        # outputs = a2m_model(inputs)
        outputs = a2v(a2m_model, a2p_model, pipe, lmk_extractor, vis, ref_image_path, audio_path, acc = True)
        labels = labels.view(-1, 1).float()  # reshape labels to match output
        
        ###### To be Modified  ( Outputs and Labels not yet sure in the same format)
        loss = criterion(outputs, labels)
        ######

        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate according to scheduler

        # Accumulate loss for the epoch
        running_loss += loss.item()

        # update tqdm progress bar
        # progress.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss 
        # epoch_loss = running_loss / len(train_loader)
        print(f"Epoch Loss: {epoch_loss:.4f}")

    print('Finished Training')

    # Save the model
    torch.save(a2m_model.state_dict(), 'a2m_model.pth')




    # # Assuming you have a Dataset class that returns (ref_image, audio_clip, ref_video)
    # dataset = YourDataset()

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # for epoch in range(100):  # Number of epochs
    #     train(model, dataloader, criterion, optimizer, device)

    #     # Save model every 10 epochs
    #     if epoch % 10 == 0:
    #         torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()