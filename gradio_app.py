import numpy as np
import cv2
import time
import torch
import supervision as sv
import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from clipaway import CLIPAwayInference

from PIL import Image
import io
import base64
from typing import List, Tuple, Optional

# YOLO-World + EfficientViT SAM imports
from masks import DeticMask, YOLOMask, HybridMask
from inference.models import YOLOWorld
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
from analyze_architecture import analyze_architectural_elements

start_init_time = time.time()
clipaway = CLIPAwayInference()
print(f"[INFO] CLIPAway initialized at {time.time() - start_init_time:.2f} seconds")

# Assume these are already created based on your current code:
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = EfficientViTSamPredictor(create_sam_model(name="xl1", weight_url="xl1.pt").to(device).eval())
detic_mask = DeticMask(sam_predictor=sam)
print(f"[INFO] DeticMask initialized at {time.time() - start_init_time:.2f} seconds")
yolo_world = YOLOWorld(model_id="yolo_world/l")
print(f"[INFO] YOLOWorld initialized at {time.time() - start_init_time:.2f} seconds")
yolo_mask_detector = YOLOMask(yolo_world, sam, confidence=0.05, nms_threshold=0.99)
print(f"[INFO] YOLOMask initialized at {time.time() - start_init_time:.2f} seconds")
hybrid_mask_detector = HybridMask(
    detic_mask=detic_mask,
    yolo_mask=yolo_mask_detector,  # Pass YOLOMask instance
    sam_predictor=sam,
    confidence=0.05,
    nms_threshold=0.99,
    overlap_threshold=0.25
)
print(f"[INFO] HybridMask initialized at {time.time() - start_init_time:.2f} seconds")

##################################################################
# 2) Demo App
##################################################################
def create_furniture_demo():
    """
    Creates the furniture detection demo interface, optionally segmenting
    the entire image with SAM if 'segment entire image' is checked.
    """
    
    def generate_number_overlay(
        image_rgb: np.ndarray, 
        masks: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        top_n: int = 10
    ) -> np.ndarray:
        """
        Generate overlay showing up to top_n largest masks, colored and numbered
        on top of the original image.
        """
        vis_image = image_rgb.copy()
        
        # Sort masks by area (descending)
        areas = [(i, np.sum(m)) for i, m in enumerate(masks)]
        sorted_indices = [idx for idx, _ in sorted(areas, key=lambda x: x[1], reverse=True)]
        top_n_indices = sorted_indices[:top_n]
        
        if colors is None:
            palette = sv.ColorPalette.DEFAULT
            colors = [palette.by_idx(i).as_rgb() for i in range(len(masks))]
        
        # First pass: draw all masks with semi-transparency
        for new_idx, orig_idx in enumerate(top_n_indices):
            mask = masks[orig_idx]
            color = colors[orig_idx]
            mask_region = vis_image[mask]
            colored_region = np.full((mask_region.shape[0], 3), color, dtype=np.uint8)
            
            alpha = 0.3  # 70% transparent overlay
            blended_region = cv2.addWeighted(
                colored_region, alpha,
                mask_region, 1 - alpha,
                0
            )
            vis_image[mask] = blended_region
            
            # Draw contour around mask
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 3)
            cv2.drawContours(vis_image, contours, -1, (0,0,0), 1)

        # Second pass: number them
        font_scale = 1.5
        thickness_outline = 5
        thickness_inner = 3
        for new_idx, orig_idx in enumerate(top_n_indices):
            mask = masks[orig_idx]
            color = colors[orig_idx]
            
            mask_uint8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                text = f"{new_idx + 1}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_inner
                )
                text_x = x + (w - text_w) // 2
                text_y = y + (h + text_h) // 2
                
                # black outline
                for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
                    cv2.putText(
                        vis_image, text, 
                        (text_x + dx, text_y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness_outline, cv2.LINE_AA
                    )
                
                # colored number
                cv2.putText(
                    vis_image, text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness_inner, cv2.LINE_AA
                )
        
        return vis_image

    def detect_furniture(
        image: np.ndarray,
        categories: str,
        confidence: float,
        nms_threshold: float,
        remove_architecture: bool,
        always_mask_categories: str,
        never_mask_categories: str,
        dilation_px: int,
        n_clipaway: int,
        seg_method: str,     # "yolo", "detic", or "hybrid"
        n_top_masks: int
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Main function that obtains masks from one of three methods:
          1. YOLO -> SAM
          2. Detic -> SAM
          3. Hybrid union (Detic + YOLO -> SAM)
        
        Then it optionally removes architecture, optionally runs CLIPAway, 
        and returns the final images + logs.
        
        Returns: 
            logs, annotated_image, unified_mask_overlay, masked_image,
            number_labeled_image, post_removal_image, clipaway_outputs
        """
        logs = []
        
        # ----------------------------------------------------------------
        # STEP 1: Get initial masks based on seg_method
        # ----------------------------------------------------------------
        start_time = time.time()
        logs.append(f"[INFO] Segmentation method: {seg_method} at {time.time() - start_time:.2f} seconds")
        
        # Optionally set YOLO categories if seg_method is YOLO or Hybrid 
        # (only YOLO currently respects user categories).
        # If detic-only, user categories not relevant for Detic. But let's do
        # it anyway for consistency; it won't affect Detic usage.
        yolo_world.set_classes([cat.strip() for cat in categories.split(",")])
        never_mask_categories = never_mask_categories.split(",")
        logs.append(f"[INFO] Never mask categories: {never_mask_categories}")
        if seg_method.lower() == "yolo":
            logs.append("[INFO] Using YOLO -> SAM.")
            all_masks = yolo_mask_detector.get_masks(image, never_mask_categories, categories)
            logs.append(f"[INFO] YOLO masks obtained at {time.time() - start_time:.2f} seconds")
        
        elif seg_method.lower() == "detic":
            logs.append("[INFO] Using Detic -> SAM.")
            all_masks = detic_mask.get_masks(image, never_mask_categories)
            logs.append(f"[INFO] Detic masks obtained at {time.time() - start_time:.2f} seconds")
        else:
            # "hybrid"
            logs.append("[INFO] Using Hybrid (Detic + YOLO) -> SAM.")
            all_masks = hybrid_mask_detector.get_hybrid_masks(image, never_mask_categories)
            logs.append(f"[INFO] Hybrid masks obtained at {time.time() - start_time:.2f} seconds")
        
        logs.append(f"[INFO] We obtained {len(all_masks)} masks total.")
        
        # ----------------------------------------------------------------
        # STEP 2: Build the "annotated_image" 
        #    - If YOLO was used, we can produce a bounding‐box overlay
        #      with class labels. If not YOLO, we simply do a mask overlay.
        # ----------------------------------------------------------------
        annotated_image = image.copy()
        if seg_method.lower() == "yolo":
            # Replicate existing YOLO bounding box annotation approach:
            results = yolo_world.infer(image, confidence=confidence)
            detections = sv.Detections.from_inference(results).with_nms(
                class_agnostic=True, threshold=nms_threshold
            )
            if len(detections.xyxy) > 0:
                # Assign masks for bounding‐box annotation (for visual debugging)
                detections.mask = np.array(all_masks)
                bgr_annot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                bgr_annot = sv.MaskAnnotator().annotate(bgr_annot, detections)
                bgr_annot = sv.BoundingBoxAnnotator().annotate(bgr_annot, detections)
                
                # Build YOLO furniture labels for display
                furniture_items = [cat.strip() for cat in categories.split(",")]
                furniture_labels = []
                for cid, conf in zip(detections.class_id, detections.confidence):
                    label_str = f"{furniture_items[cid]}: {conf:.2f}"
                    furniture_labels.append(label_str)
                
                bgr_annot = sv.LabelAnnotator().annotate(bgr_annot, detections, labels=furniture_labels)
                annotated_image = cv2.cvtColor(bgr_annot, cv2.COLOR_BGR2RGB)
            else:
                logs.append("[INFO] YOLO found no bounding boxes, so annotated_image is just the original.")
        
        else:
            # For Detic or Hybrid, we won't do bounding box annotation.
            # Instead, do a simple overlay for illustration:
            logs.append("[INFO] Annotating via mask overlay only (no bounding boxes).")
            # Create a unified mask overlay for illustration
            if len(all_masks) > 0:
                overlay_demo = image.copy()
                unify_demo = np.zeros(image.shape[:2], dtype=bool)
                for m in all_masks:
                    unify_demo |= m
                demo_overlay = np.zeros_like(image)
                demo_overlay[unify_demo] = [255,255,255]
                cv2.addWeighted(demo_overlay, 0.5, overlay_demo, 1.0, 0, overlay_demo)
                annotated_image = overlay_demo
        logs.append(f"[INFO] Annotated image obtained at {time.time() - start_time:.2f} seconds")
        
        # ----------------------------------------------------------------
        # STEP 3: Build unified_mask_overlay and masked_image. Make sure to remove never_mask_categories
        # ----------------------------------------------------------------
        unified_mask_overlay = image.copy()
        masked_image = image.copy()
        if len(all_masks) > 0:
            unify = np.zeros(image.shape[:2], dtype=bool)
            for m in all_masks:
                unify |= m
            overlay_mask = np.zeros_like(image)
            overlay_mask[unify] = [255,255,255]
            cv2.addWeighted(overlay_mask, 0.5, unified_mask_overlay, 1.0, 0, unified_mask_overlay)
            masked_image[unify] = [255,255,255]
        logs.append(f"[INFO] Unified mask overlay obtained at {time.time() - start_time:.2f} seconds")
        
        # ----------------------------------------------------------------
        # STEP 4: Generate the number-labeled image up to n_top_masks
        # ----------------------------------------------------------------
        if len(all_masks) == 0:
            logs.append("[INFO] No masks => skipping numeric-labeled image.")
            number_labeled_image = np.zeros((1,1,3),dtype=np.uint8)
        else:
            # Just label them generically, or you can add specific class labels
            # if seg_method == yolo. We'll keep it consistent:
            mask_labels = [f"mask_{i}" for i in range(len(all_masks))]
            palette = sv.ColorPalette.DEFAULT
            mask_colors = [palette.by_idx(i).as_rgb() for i in range(len(all_masks))]
            
            number_labeled_image = generate_number_overlay(
                image_rgb=image, 
                masks=all_masks, 
                labels=mask_labels, 
                colors=mask_colors,
                top_n=n_top_masks
            )
        logs.append(f"[INFO] Number-labeled image obtained at {time.time() - start_time:.2f} seconds")
        
        # ----------------------------------------------------------------
        # STEP 5: If remove_architecture => analyze with LLM
        # ----------------------------------------------------------------
        post_removal_image = np.zeros((1,1,3),dtype=np.uint8)
        new_unified = None
        if remove_architecture:
            logs.append("[INFO] Attempting architectural and category removal.")
            if len(all_masks) == 0:
                logs.append("[WARN] No masks => nothing to remove.")
            else:
                # Sort indices by area
                areas = [(i, np.sum(m)) for i, m in enumerate(all_masks)]
                sorted_indices = [idx for idx, _ in sorted(areas, key=lambda x: x[1], reverse=True)]
                logs.append(f"[INFO] Sorted indices: {sorted_indices}")

                # Call LLM
                pil_label_img = Image.fromarray(number_labeled_image)
                arch_numbers, reason = analyze_architectural_elements(pil_label_img)
                logs.append(f"[INFO] Architectural numbers from Gemini: {arch_numbers}, reason: {reason}")
                
                # If user wants to remove "never_mask_categories", e.g. fireplace, etc.
                # (You had logic for YOLO class labels, but let's keep it minimal here
                #  to preserve your existing approach. If you want to integrate that logic,
                #  just replicate it here.)
                removed_numbers = set()
                # ... (We won't replicate your entire logic unless needed)

                # Combine architecture numbers with removed numbers
                numbers_to_remove = list(set(arch_numbers) | removed_numbers)
                logs.append(f"[INFO] Final numbers to remove: {numbers_to_remove}")

                # Convert from 1-based visualization to actual all_masks indices
                removal_indices = []
                for x in numbers_to_remove:
                    # If x is within top_n, we use sorted_indices
                    # else it won't exist
                    if 1 <= x <= len(sorted_indices):
                        real_idx = sorted_indices[int(x) - 1]
                        removal_indices.append(real_idx)
                
                new_masks = [m for i, m in enumerate(all_masks) if i not in removal_indices]
                logs.append(f"[INFO] After removal, we have {len(new_masks)} masks.")
                
                # unify them
                new_unified = np.zeros(image.shape[:2], dtype=bool)
                for nm in new_masks:
                    new_unified |= nm

                # Convert boolean => uint8 => dilate
                new_unified = (new_unified.astype(np.uint8) * 255)
                kernel = np.ones((dilation_px, dilation_px), np.uint8)
                new_unified = cv2.dilate(new_unified, kernel)
                
                # Overlay
                post_removal_image = image.copy()
                new_overlay = np.zeros_like(image)
                new_overlay[new_unified > 0] = [255, 255, 255]
                cv2.addWeighted(new_overlay, 0.5, post_removal_image, 1.0, 0, post_removal_image)
        else:
            logs.append("[INFO] Architectural removal not requested.")
        logs.append(f"[INFO] Post-removal image obtained at {time.time() - start_time:.2f} seconds")
        
        # ----------------------------------------------------------------
        # STEP 6: CLIPAway
        # ----------------------------------------------------------------
        clipaway_outputs = []
        if n_clipaway > 0:
            logs.append(f"[INFO] Running CLIPAway with {n_clipaway} generations...")
            try:
                # Convert numpy image to PIL
                pil_image = Image.fromarray(image)
                
                # Make a binary mask from new_unified or from all_masks
                if new_unified is None:
                    # If no new_unified, build union from all_masks for CLIPAway
                    if len(all_masks) == 0:
                        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    else:
                        union = np.zeros(image.shape[:2], dtype=bool)
                        for m in all_masks:
                            union |= m
                        binary_mask = (union.astype(np.uint8) * 255)
                else:
                    binary_mask = new_unified.copy()  # already uint8
                
                pil_mask = Image.fromarray(binary_mask, mode='L')
                
                # Run CLIPAway inference
                clipaway_outputs = clipaway(
                    image=pil_image,
                    mask=pil_mask,
                    prompt="",
                    n_generations=n_clipaway
                )
                
                clipaway_outputs = [np.array(img) for img in clipaway_outputs]
                logs.append(f"[INFO] Successfully generated {len(clipaway_outputs)} variations")
                
            except Exception as e:
                logs.append(f"[ERROR] CLIPAway generation failed: {str(e)}")
                clipaway_outputs = []
        logs.append(f"[INFO] CLIPAway outputs obtained at {time.time() - start_time:.2f} seconds")
        
        # ----------------------------------------------------------------
        # Return everything
        # ----------------------------------------------------------------
        return (
            "\n".join(logs),
            annotated_image,
            unified_mask_overlay,
            masked_image,
            number_labeled_image,
            post_removal_image,
            clipaway_outputs
        )

    with gr.Blocks() as demo:
        gr.Markdown("""
        # 🏠 Furniture Detection Demo
        This demo can:
         1. Use **Detic**, **YOLO**, or **Hybrid** segmentation with SAM.
         2. Optionally call Gemini to identify permanent architecture to remove from the mask.
         3. Optionally run CLIPAway for inpainted variations.
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Room Image")
                
                seg_method = gr.Radio(
                    choices=["yolo", "detic", "hybrid"],
                    value="yolo",
                    label="Segmentation Method"
                )
                
                categories = gr.Textbox(
                    label="YOLO / Hybrid Categories",
                    value="sofa, couch, armchair, candle, sculpture, dining chair, office chair, stool, ottoman, bench, rocking chair, piano, clothing, bean bag, recliner, coffee table, dining table, side table, end table, console table, desk, nightstand, tv stand, vanity table, bookshelf, bookcase, cabinet, dresser, wardrobe, closet, chest of drawers, display cabinet, media center, storage bench, sideboard, buffet, bed frame, headboard, footboard, mattress, daybed, futon, floor lamp, table lamp, ceiling light, chandelier, wall sconce, pendant light, track lighting, reading light, painting, cushion, flowers, book, mirror, carpet, vase, bowl, food, water jug, guitar, framed image, flower pot, plant, fireplace, tv, vase, bottle, basket, plate, clock.",
                    lines=3,
                )
                always_mask_categories = gr.Textbox(
                    label="Always Mask Categories",
                    value="framed image",
                    lines=3,
                )
                never_mask_categories = gr.Textbox(
                    label="Never Mask Categories",
                    value="Fireplace, carpet",
                    lines=3,
                )

                n_clipaway = gr.Slider(0,10,5,step=1,label="CLIPAway Generations")
                dilation_px = gr.Slider(0,100,20,step=1,label="Dilation Pixels")
                n_top_masks = gr.Slider(1, 30, 10, step=1, label="Number of Masks to Pass to Architecture Removal")

                with gr.Row():
                    confidence = gr.Slider(0,1,0.05,step=0.01,label="Detection Confidence")
                    nms_threshold = gr.Slider(0,1,0.99,step=0.01,label="NMS Threshold")

                remove_arch_cb = gr.Checkbox(value=True, label="Remove Architecture w/ Gemini?")
                
                run_btn = gr.Button("🔍 Run")

        logs_out = gr.Textbox(label="Logs", lines=10, interactive=False)
        
        with gr.Row():
            out_annotated = gr.Image(label="Detailed Detection")
            out_unified = gr.Image(label="Unified Overlay")
            out_masked = gr.Image(label="Masked Furniture")
        with gr.Row():
            out_lettered = gr.Image(label="Numbered Segments")
            out_removed = gr.Image(label="Post-Removal Image")
        with gr.Row():
            clipaway_gallery = gr.Gallery(
                label="CLIPAway Generations", 
                columns=4,
                height="auto"
            )

        # Hook up button
        run_btn.click(
            fn=detect_furniture,
            inputs=[
                input_image, 
                categories, 
                confidence, 
                nms_threshold,
                remove_arch_cb, 
                always_mask_categories, 
                never_mask_categories, 
                dilation_px, 
                n_clipaway,
                seg_method,
                n_top_masks
            ],
            outputs=[
                logs_out, 
                out_annotated, 
                out_unified, 
                out_masked, 
                out_lettered, 
                out_removed, 
                clipaway_gallery
            ],
            queue=True
        )

    return demo


web_app = FastAPI()
demo = create_furniture_demo()
demo.queue(max_size=5)  # Same Gradio queue usage
app = mount_gradio_app(app=web_app, blocks=demo, path="/")

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
