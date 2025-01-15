import pydantic
import pathlib
import shlex
import subprocess
import os
import modal
import dotenv
import urllib.request

GRADIO_PORT = 8000
app = modal.App("gradio-app-vit")

dotenv.load_dotenv(override=True)

def download_efficientvit_model(dest_path: str = '/root/'):
    """Download EfficientViT SAM model if not present."""
    EFFICIENTVIT_SAM_URL = "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main"
    MODEL_FILE = "efficientvit_sam_xl1.pt"
    DEST_FILE = "xl1.pt"
    os.makedirs(dest_path, exist_ok=True)
    model_path = os.path.join(dest_path, DEST_FILE)
    
    if not os.path.exists(model_path):
        print(f"Downloading EfficientViT SAM model to {model_path}...")
        url = f"{EFFICIENTVIT_SAM_URL}/{MODEL_FILE}"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")
    else:
        print("EfficientViT SAM model already exists, skipping download")
    
    return model_path

def clone_clipaway_repo():
    import os
    os.system("git clone https://github.com/YigitEkin/CLIPAway.git")
    os.system("git clone https://github.com/SunzeY/AlphaCLIP.git")
    
def download_checkpoints():
    import os
    import subprocess
    # Import all the stable diffusion models we will use later.
    from diffusers import StableDiffusionInpaintPipeline
    import torch
    
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "botp/stable-diffusion-v1-5-inpainting",
        safety_checker=None,
        torch_dtype=torch.float32
    )
    # Cache to a directory that persists in the image
    pipeline.save_pretrained("/root/sd_models/stable-diffusion-v1-5-inpainting")
    
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        safety_checker=None,
        torch_dtype=torch.float32
    )
    pipeline.save_pretrained("/root/sd_models/stable-diffusion-2-inpainting")
    
    # Create directory structure
    os.makedirs("/root/ckpts/AlphaCLIP", exist_ok=True)
    os.makedirs("/root/ckpts/IPAdapter/image_encoder", exist_ok=True)
    os.makedirs("/root/ckpts/CLIPAway", exist_ok=True)
    
    print("Downloading Alpha-CLIP weights...")
    subprocess.run(["gdown", "1JfzOTvjf0tqBtKWwpBJtjYxdHi-06dbk"], 
                  cwd="/root/ckpts/AlphaCLIP")
    
    print("Downloading IP-Adapter weights...")
    ip_adapter_files = {
        "ip-adapter_sd15.bin": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin",
        "image_encoder/pytorch_model.bin": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin",
        "image_encoder/config.json": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json"
    }
    for filename, url in ip_adapter_files.items():
        output_path = os.path.join("/root/ckpts/IPAdapter", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run(["wget", "-O", output_path, url])
    
    print("Downloading CLIPAway weights...")
    subprocess.run(["gdown", "1lFHAT2dF5GVRJLxkF1039D53gixHXaTx"],
                  cwd="/root/ckpts/CLIPAway")
    
    print("Finished downloading pretrained models.")

image = (
    modal.Image.micromamba(python_version="3.10")
    .apt_install("git", "python3-opencv", "libgl1-mesa-glx", 
        "git", "wget", "curl", "build-essential", "libglib2.0-0", "libsm6", "fonts-open-sans",
        "libxext6", "libxrender-dev", "libgl1-mesa-glx", "libffi-dev",
        "libssl-dev", "libbz2-dev", "libreadline-dev", "libsqlite3-dev",
        "libncurses5-dev", "libgomp1", "libuuid1", "ca-certificates",
        "libstdc++6", "libgcc1", "libopenblas-dev", "git-lfs"
    )
    .micromamba_install(
        [
            "_libgcc_mutex=0.1=main",
            "_openmp_mutex=5.1=1_gnu",
            "bzip2=1.0.8",
            "ca-certificates",
            "ld_impl_linux-64",
            "libffi",
            "libgcc-ng",
            "libgomp",
            "libstdcxx-ng",
            "libuuid",
            "ncurses",
            "openssl",
            "pip",
            "readline",
            "setuptools",
            "sqlite",
            "tk",
            "wheel",
            "xz",
            "zlib",
            "anthropic",
            "backoff",
            "python-dotenv",
        ],
        channels=["conda-forge", "defaults"]
    )
    .pip_install(
        "timm",
        "absl-py",
        "accelerate==0.28.0",
        "anthropic",
        "backoff==2.2.1",  # Keeping explicit version over unversioned
        "clean-fid",
        "diffusers==0.29.0",
        "einops==0.7.0",
        "fastapi==0.112.4",
        "ftfy==6.2.0",
        "gdown==5.1.0",
        "git+https://github.com/facebookresearch/segment-anything.git",
        "git+https://github.com/openai/CLIP.git",
        "git+https://github.com/tencent-ailab/IP-Adapter.git",
        "google-generativeai==0.8.3",  # Keeping versioned over unversioned
        "gradio",
        "inference[yolo-world]",
        "jsonschema==4.19.1",
        "loralib",
        "numpy",
        "omegaconf==2.3.0",
        "onnx==1.15.0",
        "onnxsim==0.4.35",
        "opencv-python",
        "pandas",
        "pillow==10.3.0",
        "pycparser==2.22",
        "pydantic",  
        "regex==2023.12.25",
        "setuptools==69.5.1",
        "supervision==0.18.0",
        "timm==0.9.12",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "tqdm==4.66.2",
        "traitlets==5.14.3",
        "transformers==4.39.3"
    )
    .run_commands(
        "git lfs install",
        "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'",
        "mkdir -p /root/third_party",
        "git clone https://github.com/xingyizhou/CenterNet2.git /root/third_party/CenterNet2",
        "git clone https://github.com/facebookresearch/Detic.git --recurse-submodules /root/Detic",
        "cd /root/Detic && pip install -r requirements.txt",
        "mkdir -p /opt/models",
    )
    .run_function(download_efficientvit_model)
    .run_function(clone_clipaway_repo)
    .run_function(download_checkpoints)
    .add_local_file(local_path='./gradio_app.py', remote_path='/root/gradio_app.py')
    .add_local_dir(
        local_path='../yolo-world-with-efficientvit-sam', 
        remote_path='/root/'
    )
)

@app.function(
    gpu="A100",
    image=image,
    allow_concurrent_inputs=100,  # Ensure we can handle multiple requests
    concurrency_limit=1,  # Ensure all requests end up on the same container
)
@modal.web_server(GRADIO_PORT, startup_timeout=120)
def web_app():
    import os
    import sys
        
    # Set up environment
    os.chdir('/root/')
    os.environ['PYTHONPATH'] = os.pathsep.join([
        os.environ.get('PYTHONPATH', ''),  # Get existing or empty string
        '/root/',
        '/root/Detic',
        '/root/third_party/CenterNet2'
    ])
    os.environ['EFFICIENTVIT_SAM_PATH'] = "/root/models/xl1.pt"
    
    # Set up CLIPAway
    sys.path.append("/root/CLIPAway")
    sys.path.append("/root/AlphaCLIP")
    sys.path.append("/root/Detic")
    sys.path.append("/root/third_party/CenterNet2")
    
    # print to console all the files in /root/
    print(f"COMPLETED SETUP. Files in /root/: {os.listdir('/root/')}.")
    
    # Launch the Gradio app
    target = shlex.quote(str('/root/gradio_app.py'))
    cmd = f"python {target} --host 0.0.0.0 --port {GRADIO_PORT}"
    subprocess.Popen(cmd, shell=True)
