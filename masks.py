from typing import List, Tuple
import numpy as np
import supervision as sv
from typing import List
from inference.models import YOLOWorld
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
import time


class DeticMask:
    """
    A minimal class that:
      1) Uses Detic to detect objects in an image, producing bounding boxes.
      2) Expands these bounding boxes by a scale factor.
      3) Uses *your* SAM predictor (EfficientViTSamPredictor or similar) to generate masks from those boxes.
    """

    def __init__(self, 
                 sam_predictor: EfficientViTSamPredictor = None,
                 box_expansion_scale: float = 1.3,
                 confidence_threshold: float = 0.31,
                 nms_threshold: float = 0.7):
        init_start_time = time.time()
        print(f"Initialization Models started")

        import torch
        import os
        import sys
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detic.config import add_detic_config
        from centernet.config import add_centernet_config

        # Add required paths
        sys.path.insert(0, '/root/third_party/CenterNet2')
        sys.path.insert(0, '/root/Detic')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize Detic
        print("Initializing Detic...")
        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        
        model_path = "/root/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        config_path = "/root/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"

        # Download model if needed
        if not os.path.exists(model_path):
            print("Downloading model for the first time...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_url = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
            self.download_file(model_url, model_path)

        # Load Detic config
        print(f"Loading Detic config from {config_path}")
        self.cfg.merge_from_file(config_path)

        # Configure Detic
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.31
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
        self.cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS = True
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "/root/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy"
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1203
        self.cfg.MODEL.DEVICE = self.device

        # Set metadata and initialize predictor
        self.cfg.DATASETS.TRAIN = ()
        self.cfg.DATASETS.TEST = ('lvis_v1_val',)
        self.cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = '/root/Detic/datasets/metadata/lvis_v1_train_cat_info.json'
        
        # Initialize Detic predictor
        self.predictor = DefaultPredictor(self.cfg)

        # Set up vocabulary
        from detectron2.data import MetadataCatalog
        from detic.modeling.utils import reset_cls_test
        self.metadata = MetadataCatalog.get('lvis_v1_val')
        classifier = "/root/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy"
        self.num_classes = len(self.metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, self.num_classes)
        print(f"Vocabulary initialized with {self.num_classes} classes")

        # Initialize SAM-HQ
        print("Initializing SAM-HQ...")
        if sam_predictor is None:
            sam = EfficientViTSamPredictor(create_sam_model(name="xl1", weight_url="xl1.pt").to("cuda").eval())
        else:
            sam = sam_predictor
        self.sam_predictor = sam
        print(f"Initialization Models completed in {time.time() - init_start_time:.2f} seconds")

        self.box_expansion_scale = box_expansion_scale
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
    def download_file(self, url: str, destination: str):
        """Download file with progress bar"""
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def expand_boxes(self, boxes: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Expand each bounding box by self.box_expansion_scale, then clamp to image boundaries.
        
        Args:
            boxes: Nx4 array of (x1, y1, x2, y2) bounding boxes.
            image_size: (width, height) of the image.
        
        Returns:
            A new Nx4 array of expanded boxes.
        """
        W, H = image_size
        scale = self.box_expansion_scale
        processed_boxes = boxes.copy()

        for i in range(len(processed_boxes)):
            x1, y1, x2, y2 = processed_boxes[i]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = (x2 - x1)
            height = (y2 - y1)

            new_width = width * scale
            new_height = height * scale

            new_x1 = center_x - (new_width / 2.0)
            new_y1 = center_y - (new_height / 2.0)
            new_x2 = center_x + (new_width / 2.0)
            new_y2 = center_y + (new_height / 2.0)

            # Clamp to image boundaries
            new_x1 = max(0, min(W, new_x1))
            new_y1 = max(0, min(H, new_y1))
            new_x2 = max(0, min(W, new_x2))
            new_y2 = max(0, min(H, new_y2))

            processed_boxes[i] = [new_x1, new_y1, new_x2, new_y2]

        return processed_boxes

    def get_masks(self, image_rgb: np.ndarray, never_mask_categories: List[str] = []) -> List[np.ndarray]:
        """
        Takes RGB image and returns list of boolean masks.
        
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB order
            
        Returns:
            List of boolean masks, each shape (H, W)
        """
        import torch
        
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected RGB image as numpy array with shape (H, W, 3)")
            
        try:
            # Convert to BGR for Detic
            image_bgr = image_rgb[:, :, ::-1].copy()  # Make a contiguous copy
            outputs = self.predictor(image_bgr)
            
            # Move to CPU safely
            instances = outputs["instances"]
            min_area = 10000  # e.g., 100 pixels
            keep = instances.pred_boxes.area() > min_area
            instances = instances[keep]
            classes = instances.pred_classes.cpu().numpy()
            class_names = [self.metadata.thing_classes[i] for i in classes]
            if hasattr(instances, 'to') and instances.pred_boxes.tensor.device.type == 'cuda':
                instances = instances.to("cpu")
            
            # If nothing is detected, return early
            if not len(instances):
                return []

            # Safely get boxes
            boxes = instances.pred_boxes.tensor
            if hasattr(boxes, 'device') and boxes.device.type == 'cuda':
                boxes = boxes.cpu()
            boxes = boxes.numpy()

            # Expand bounding boxes
            expanded_boxes = self.expand_boxes(boxes, (image_rgb.shape[1], image_rgb.shape[0]))

            # Prepare SAM
            self.sam_predictor.set_image(image_rgb, image_format="RGB")

            # Generate masks
            all_masks = []
            for i, box in enumerate(expanded_boxes):
                category = class_names[i].lower()
                try:
                    should_skip = any(
                        never_cat.lower() in category 
                        for never_cat in never_mask_categories
                    )
                    if should_skip:
                        continue

                    mask_tensor, _, _ = self.sam_predictor.predict(
                        box=box,
                        multimask_output=False
                    )
                    # Handle the mask tensor properly depending on its type
                    if torch.is_tensor(mask_tensor):
                        if mask_tensor.device.type == 'cuda':
                            mask_tensor = mask_tensor.cpu()
                        mask_np = mask_tensor.squeeze().numpy()
                    else:
                        # If it's already a numpy array
                        mask_np = mask_tensor.squeeze()
                    
                    mask_np = mask_np.astype(bool)
                    all_masks.append(mask_np)
                except Exception as e:
                    print(f"Error generating mask for box {box}: {e}")
                    continue

            return all_masks

        except Exception as e:
            print(f"Error in get_masks: {e}")
            return []
            
        finally:
            # Cleanup
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

class YOLOMask:
    """
    YOLOMask detects objects in an image using YOLOWorld and then generates
    segmentation masks for each detected bounding box using your SAM predictor.
    
    Attributes:
        yolo_world (YOLOWorld): The YOLO detector instance.
        sam (EfficientViTSamPredictor): The SAM predictor instance.
        confidence (float): Confidence threshold used when calling YOLOWorld.
        nms_threshold (float): The non-maximum suppression threshold.
    """
    def __init__(
        self,
        yolo_world: YOLOWorld,
        sam: EfficientViTSamPredictor,
        confidence: float = 0.05,
        nms_threshold: float = 0.99,
    ):
        """
        Initialize the YOLOMask detector.

        Args:
            yolo_world (YOLOWorld): An instance of your YOLO detector.
            sam (EfficientViTSamPredictor): An instance of your SAM predictor.
            confidence (float, optional): Detection confidence threshold. Defaults to 0.05.
            nms_threshold (float, optional): NMS threshold. Defaults to 0.99.
        """
        self.yolo_world = yolo_world
        self.sam = sam
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        
    def get_masks(self, image_rgb: np.ndarray, never_mask_categories: List[str] = [], categories: List[str] = []) -> List[np.ndarray]:
        """
        Generates segmentation masks from an image using a YOLO->SAM pipeline.
        
        Args:
            image_rgb (np.ndarray): Input image (in RGB order) as a NumPy array.
            never_mask_categories (List[str]): Categories to exclude from masking.

        Returns:
            List[np.ndarray]: A list of boolean masks.
        """
        # Run YOLOWorld inference with the given confidence
        results = self.yolo_world.infer(image_rgb, confidence=self.confidence)
        # Convert detections using supervision and apply NMS
        detections = sv.Detections.from_inference(results).with_nms(
            class_agnostic=True, threshold=self.nms_threshold
        )

        if len(detections.xyxy) == 0:
            return []
        self.sam.set_image(image_rgb, image_format="RGB")

        # Get categories from detection results
        masks = []
        for cid, conf, xyxy in zip(detections.class_id, detections.confidence, detections.xyxy):
            # Get category from the detection class ID
            label_str = categories[cid]
            print(f"Label: {label_str} with confidence {conf} and xyxy {xyxy}")
            if label_str in never_mask_categories:
                print(f"Skipping category {label_str} because it's in never_mask_categories {never_mask_categories}")
                continue
            # generate mask for this box
            mask, _, _ = self.sam.predict(box=xyxy, multimask_output=False)
            mask = mask.squeeze()
            masks.append(mask)
            
        return masks
        


class HybridMask:
    """
    This class combines Detic-based mask generation with YOLO-based mask generation,
    using your existing SAM predictor in both cases. The final “hybrid” mask is produced
    by taking the union of all masks (logical OR).
    """
    def __init__(
        self,
        detic_mask: DeticMask,  # An instance of DeticMask
        yolo_mask: YOLOMask,    # Changed from yolo_world to yolo_mask
        sam_predictor,
        confidence: float = 0.05,
        nms_threshold: float = 0.99,
        overlap_threshold: float = 0.5
    ):
        self.detic_mask = detic_mask
        self.yolo_mask = yolo_mask  # Store YOLOMask instance
        self.sam = sam_predictor
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.overlap_threshold = overlap_threshold

    def get_hybrid_masks(self, image_rgb: np.ndarray) -> List[np.ndarray]:
        """
        Returns a hybrid list of masks, merging those that overlap significantly.
        Masks that overlap by more than 50% IoU are merged into a single mask.
        """
        # Get masks from both detectors
        detic_masks = self.detic_mask.get_masks(image_rgb)
        yolo_masks = self.yolo_mask.get_masks(image_rgb)
        
        # Combine all masks into initial list
        all_masks = detic_masks + yolo_masks
        
        if len(all_masks) == 0:
            return []
        
        def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
            """Calculate Intersection over Union between two masks."""
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            return intersection / union if union > 0 else 0.0

        def merge_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
            """Merge two masks using logical OR."""
            return np.logical_or(mask1, mask2)

        # Keep track of which masks need to be merged
        merged_masks = []
        masks_to_process = all_masks.copy()
        
        while masks_to_process:
            current_mask = masks_to_process.pop(0)
            masks_to_merge = [current_mask]
            
            # Find all masks that overlap significantly with current_mask
            i = 0
            while i < len(masks_to_process):
                iou = calculate_iou(current_mask, masks_to_process[i])
                if iou > self.overlap_threshold:  # 50% overlap threshold
                    masks_to_merge.append(masks_to_process.pop(i))
                else:
                    i += 1
            
            # Merge all overlapping masks
            final_mask = masks_to_merge[0]
            for mask in masks_to_merge[1:]:
                final_mask = merge_masks(final_mask, mask)
            
            merged_masks.append(final_mask)
        
        return merged_masks


