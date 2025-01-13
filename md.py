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

def download_efficientvit_model(dest_path: str = '/root/yolo-world-with-efficientvit-sam/'):
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

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "python3-opencv", "libgl1-mesa-glx")
    .pip_install(
        "inference[yolo-world]==0.9.13",
        "supervision==0.18.0",
        "fastapi==0.112.4", 
        "pydantic",
        "gradio==4.26.0",
        "timm==0.9.12",
        "onnx==1.15.0",
        "onnxsim==0.4.35",
        "backoff==2.2.1",
        "jsonschema==4.19.1",
        "google-generativeai==0.8.3",
        "git+https://github.com/facebookresearch/segment-anything.git",
    )
    .run_function(download_efficientvit_model)
    .add_local_file(local_path='./gradio_app.py', remote_path='/root/gradio_app.py')
    .add_local_dir(
        local_path='../yolo-world-with-efficientvit-sam', 
        remote_path='/root/yolo-world-with-efficientvit-sam'
    )
)

@app.function(
    gpu="T4",
    image=image,
    allow_concurrent_inputs=100,  # Ensure we can handle multiple requests
    concurrency_limit=1,  # Ensure all requests end up on the same container
)
@modal.web_server(GRADIO_PORT, startup_timeout=60)
def web_app():
    import os
        
    # Set up environment
    os.chdir('/root/yolo-world-with-efficientvit-sam')
    os.environ['PYTHONPATH'] = '/root/yolo-world-with-efficientvit-sam'
    os.environ['EFFICIENTVIT_SAM_PATH'] = "/root/yolo-world-with-efficientvit-sam/models/xl1.pt"
    
    # Launch the Gradio app
    target = shlex.quote(str('/root/gradio_app.py'))
    cmd = f"python {target} --host 0.0.0.0 --port {GRADIO_PORT}"
    subprocess.Popen(cmd, shell=True)

class GenerateRoomRequest(pydantic.BaseModel):
    image_base64: str
    image_url: str
    
# @app.cls(
#     gpu="T4",
#     image=image,
#     container_idle_timeout=120,
#     allow_concurrent_inputs=10,
#     keep_warm=0,
#     concurrency_limit=20,
#     timeout=600
# )
# class GenerateMask:
#     @modal.enter()
#     def initialize_models(self):
#         import torch
#         from inference.models import YOLOWorld
#         from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
#         from efficientvit.sam_model_zoo import create_sam_model
#         self.yolo_world = YOLOWorld(model_id="yolo_world/l")
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.sam = EfficientViTSamPredictor(
#             create_sam_model(name="xl1", weight_url="xl1.pt").to(device).eval()
#         )

#     def decode_image(self, image_b64: str):
#         import base64
#         import numpy as np
#         import cv2
#         """Decode base64 image to numpy array."""
#         img_data = base64.b64decode(image_b64.split(',')[1] if ',' in image_b64 else image_b64)
#         nparr = np.frombuffer(img_data, np.uint8)
#         return cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

#     def encode_mask(self, mask) -> str:
#         import base64
#         import numpy as np
#         import cv2
#         """Encode binary mask to base64 PNG."""
#         _, encoded = cv2.imencode('.png', mask)
#         return base64.b64encode(encoded).decode('utf-8')

#     @modal.web_endpoint(method="POST")
#     def detect_furniture(self, request: GenerateRoomRequest):
#         import supervision as sv
#         import numpy as np
#         prompt = "sofa, couch, armchair, dining chair, office chair, stool, ottoman, bench, rocking chair, bean bag, recliner, coffee table, dining table, side table, end table, console table, desk, nightstand, tv stand, vanity table, bookshelf, bookcase, cabinet, dresser, wardrobe, closet, chest of drawers, display cabinet, media center, storage bench, sideboard, buffet, bed frame, headboard, footboard, mattress, daybed, futon, floor lamp, table lamp, ceiling light, chandelier, wall sconce, pendant light, track lighting, reading light, painting, cushion, flowers, book, mirror, carpet, vase, bowl, food"
        
#         # Decode input image
#         image = self.decode_image(request.image_base64)
        
#         # Set up detection classes
#         self.yolo_world.set_classes(request.categories.split(","))
        
#         # Run YOLO detection
#         results = self.yolo_world.infer(image, confidence=request.threshold)
#         detections = sv.Detections.from_inference(results).with_nms(
#             class_agnostic=True,
#             threshold=request.nms_threshold
#         )

#         # Initialize empty mask
#         combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
#         if len(detections.xyxy) > 0:
#             # Generate and combine masks
#             self.sam.set_image(image, image_format="RGB")
#             for xyxy in detections.xyxy:
#                 mask, _, _ = self.sam.predict(box=xyxy, multimask_output=False)
#                 combined_mask |= mask.squeeze().astype(np.uint8) * 255

#         # Return base64 encoded PNG of binary mask
#         return {"mask": self.encode_mask(combined_mask)}