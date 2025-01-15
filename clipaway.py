import sys
import os
import random
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from omegaconf import OmegaConf
from diffusers import StableDiffusionInpaintPipeline

# Add CLIPAway paths
sys.path.append("/root/CLIPAway")
sys.path.append("/root/AlphaCLIP")

from model.clip_away import CLIPAway

def pad_image(image: Image.Image, size: int = 512) -> Image.Image:
    original_width, original_height = image.size
    aspect = original_width / original_height

    if aspect > 1:  # width > height
        new_width = size
        new_height = int(size / aspect)
        padding_top = (size - new_height) // 2
        padding_bottom = size - new_height - padding_top
        padding_left = 0
        padding_right = 0
    else:  # height > width
        new_height = size
        new_width = int(size * aspect)
        padding_left = (size - new_width) // 2
        padding_right = size - new_width - padding_left
        padding_top = 0
        padding_bottom = 0
    
    # Resize maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Choose a background color based on the image mode
    if image.mode == "L":
        # For grayscale, use a single integer (0 for black)
        bg_color = 0
    else:
        # For modes like RGB, use a tuple (0, 0, 0)
        bg_color = (0, 0, 0)
    
    # Create new square image with padding
    result = Image.new(image.mode, (size, size), bg_color)
    result.paste(image, (padding_left, padding_top))
    
    return result

class CLIPAwayInference:
    def __init__(
        self,
        config_path: str = "/root/CLIPAway/config/inference_config.yaml",
        sd_model_path: str = "/root/sd_models/stable-diffusion-v1-5-inpainting",
        device: str = "cuda"
    ):
        """Initialize CLIPAway inference pipeline.
        
        Args:
            config_path: Path to CLIPAway config YAML
            sd_model_path: Path to SD model checkpoint
            device: Device to run inference on
        """
        # Load config
        self.cfg = OmegaConf.load(config_path)
        self.device = device
        
        # Initialize SD pipeline
        self.sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_path,
            safety_checker=None,
            torch_dtype=torch.float32
        ).to(device)
        
        # Initialize CLIPAway
        self.clipaway = CLIPAway(
            sd_pipe=self.sd_pipeline,
            image_encoder_path=self.cfg.image_encoder_path,
            ip_ckpt=self.cfg.ip_adapter_ckpt_path,
            alpha_clip_path=self.cfg.alpha_clip_ckpt_pth, 
            config=self.cfg,
            alpha_clip_id=self.cfg.alpha_clip_id,
            device=device,
            num_tokens=4
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        n_generations: int = 1,
        scale: float = 1.0,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """Run CLIPAway inference with proper padding and mask handling."""
        if seed is None:
            seed = random.randint(0, 999999)
            
        # Pad images to square maintaining aspect ratio
        size = 512  # Standard size for SD
        image = pad_image(image, size)
        mask = pad_image(mask, size)
            
        # Ensure proper mask format (binary black and white)
        if mask.mode != 'L':
            mask = mask.convert('L')
            
        # Convert to numpy, ensure binary mask
        mask_array = np.array(mask)
        binary_mask = np.zeros_like(mask_array, dtype=np.uint8)
        binary_mask[mask_array > 127] = 255
        mask = Image.fromarray(binary_mask)
            
        with torch.no_grad():
            try:
                outputs = self.clipaway.generate(
                    pil_image=[image],
                    alpha=[mask],
                    prompt=prompt,
                    scale=scale,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    num_images_per_prompt=n_generations
                )
                return outputs
            except Exception as e:
                print(f"Error during CLIPAway generation: {str(e)}")
                return []

# Initialize CLIPAway inference pipeline