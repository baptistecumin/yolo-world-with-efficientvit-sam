import numpy as np
import cv2
import torch
import supervision as sv
import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

from PIL import Image, ImageDraw, ImageFont
import io
import base64
import requests

import backoff
from json.decoder import JSONDecodeError
from typing import List, Tuple, Optional

import google.generativeai as genai
genai.configure(api_key="AIzaSyAcJ9vmQhxzI_aCFwIxmedybQF6NRUlecY")

# YOLO-World + EfficientViT SAM imports
from inference.models import YOLOWorld
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model


##################################################################
# 1) Architectural Analysis - Gemini
##################################################################
def analyze_architectural_elements(
    labeled_image: Image.Image,
    model: str = "gemini-1.5-flash",
    prompt: str = (
        "You are an interior design expert analyzing a room image with numbered regions. "
        "Your task is to identify which numbered elements are PERMANENT architectural features "
        "that would require construction work to remove (e.g., built-in cabinets, walls, windows, "
        "fireplaces, built-in shelving). \n\n"
        "Important guidelines:\n"
        "- If an element can be removed without construction work, it is considered movable furniture\n"
        "- ALL artwork and wall decorations are considered movable, even if mounted\n"
        "- ALL light fixtures are considered movable, including ceiling lights and sconces\n"
        "- Built-in cabinets, fireplaces, and permanent shelving ARE architectural features\n"
        "- If you are unsure about an element, consider it movable\n\n"
        "Use the select_architectural_elements tool to specify ONLY the numbers that correspond "
        "to permanent, non-movable architectural features. Only include numbers you are 100% certain "
        "represent elements that require construction to remove."
    )
) -> Tuple[List[int], str]:
    """
    Uses Gemini to determine which numbered masks represent permanent architectural
    features that would require construction to remove, excluding all movable furniture
    and decorative elements.
    
    Returns: (numbers_to_remove, explanation)
    """
    architectural_tool = {
        'function_declarations': [{
            'name': 'select_architectural_elements',
            'description': (
                "Select which numbered regions represent permanent architectural features "
                "that would require construction work to remove. Only include numbers you "
                "are 100% certain about."
            ),
            'parameters': {
                'type': 'OBJECT',
                'properties': {
                    'explanation': {
                        'type': 'STRING',
                        'description': (
                            "Detailed explanation of which items are permanent architectural "
                            "features and why they require construction to remove."
                        ),
                    },
                    'numbers_to_remove': {
                        'type': 'array',
                        'description': (
                            "List of integer IDs corresponding to permanent architectural "
                            "elements that require construction to remove."
                        ),
                        'items': {
                            'type': 'number',
                        },
                    },
                },
                'required': ['numbers_to_remove', 'explanation']
            }
        }]
    }

    # Convert the labeled PIL image to base64
    labeled_image = labeled_image.resize((512, 512))  # clamp size for consistency
    buf = io.BytesIO()
    labeled_image.save(buf, format="JPEG")
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    parts = [
        {'mime_type': 'image/jpeg', 'data': b64_image},
        prompt
    ]

    try:
        from google.generativeai import GenerativeModel
        model_instance = GenerativeModel(model_name=model, tools=architectural_tool)
        chat = model_instance.start_chat()
        response = chat.send_message(parts)
        print("[Architectural] Response from Gemini:\n", response)
        
        # Parse function call
        for candidate in response.candidates:
            if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                continue
            for part in candidate.content.parts:
                if not hasattr(part, 'function_call'):
                    continue
                fn_call = part.function_call
                if not hasattr(fn_call, 'args'):
                    continue
                args_dict = fn_call.args
                if args_dict is None:
                    continue
                numbers = list(args_dict.get('numbers_to_remove', []))
                explanation = str(args_dict.get('explanation', ''))
                return (numbers, explanation)

        print("[Architectural] No valid function call found in response.")
        return ([], "No valid function call found in Gemini response.")

    except Exception as e:
        print("[Architectural] Error calling Gemini:", e)
        raise

    return ([], "Unexpected error occurred.")

##################################################################
# 2) Demo App
##################################################################
def create_furniture_demo():
    """
    Creates the furniture detection demo interface, optionally segmenting
    the entire image with SAM if 'segment entire image' is checked.
    """
    # Initialize YOLO-World and SAM
    yolo_world = YOLOWorld(model_id="yolo_world/l")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = EfficientViTSamPredictor(
        create_sam_model(name="xl1", weight_url="xl1.pt").to(device).eval()
    )
    
    def generate_number_overlay(
        image_rgb: np.ndarray, 
        masks: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Generate overlay showing the top 10 largest masks colored and numbered
        on top of the original image.
        """
        # Start with original image
        vis_image = image_rgb.copy()
        
        # Sort masks by area and get top 10
        areas = [(i, np.sum(m)) for i, m in enumerate(masks)]
        sorted_indices = [idx for idx, _ in sorted(areas, key=lambda x: x[1], reverse=True)]
        top_10_indices = sorted_indices[:10]
        
        # If no colors provided, generate them using supervision's color palette
        if colors is None:
            palette = sv.ColorPalette.DEFAULT
            colors = [palette.by_idx(i).as_rgb() for i in range(len(masks))]
        
        # First pass: draw all masks with semi-transparency
        for new_idx, orig_idx in enumerate(top_10_indices):
            mask = masks[orig_idx]
            color = colors[orig_idx]
                
            # Create a mask-sized slice for blending
            mask_region = vis_image[mask]
            colored_region = np.full((mask_region.shape[0], 3), color, dtype=np.uint8)
            
            # Blend only the masked region. Make it 70% transparent
            alpha = 0.3
            blended_region = cv2.addWeighted(
                colored_region, alpha,
                mask_region, 1 - alpha,
                0
            )
            
            # Put the blended region back
            vis_image[mask] = blended_region
            
            # Draw thick contour around mask
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color, 3)  # Draw colored contour
            cv2.drawContours(vis_image, contours, -1, (0,0,0), 1)  # Draw thin black outline

        # Second pass: draw numbers
        font_scale = 1.5
        thickness_outline = 5
        thickness_inner = 3

        for new_idx, orig_idx in enumerate(top_10_indices):
            mask = masks[orig_idx]
            color = colors[orig_idx]
            
            # Convert mask to uint8 for contour finding
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Find contours to place number
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate text size for centering
                text = f"{new_idx + 1}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_inner
                )
                
                # Calculate centered position
                text_x = x + (w - text_w) // 2
                text_y = y + (h + text_h) // 2
                
                # Draw black outline for contrast (8-directional)
                for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
                    cv2.putText(
                        vis_image, text, 
                        (text_x + dx, text_y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness_outline, cv2.LINE_AA
                    )
                
                # Draw colored number matching the mask
                cv2.putText(
                    vis_image, text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness_inner, cv2.LINE_AA
                )
        
        return vis_image
        
    def entire_image_segmentation(img: np.ndarray) -> List[np.ndarray]:
        """
        Example "segment entire image" logic. 
        We'll do a naive approach: sample a grid of points or something similar.
        For demonstration, let's do a single 'everything' mask.

        A more realistic approach would be to call the SAM "predict everything"
        method if your library or code has it (like a tile-based approach).
        """
        # Just do a single mask = entire image for demonstration:
        # In real usage, you'd do something more advanced.
        # We'll assume we get multiple masks in a real approach.
        # For demonstration, let's produce ~2 random masks (just as placeholders).
        h, w = img.shape[:2]
        # For an actual "predict everything," you may call something like:
        # all_masks = sam.predict_tiled(img, tile_size=256, overlap=64, etc.)
        # and return all_masks. We'll do a toy approach here:
        mask1 = np.zeros((h, w), dtype=bool)
        mask2 = np.zeros((h, w), dtype=bool)
        cv2.circle(mask1, (w//3, h//3), min(h,w)//4, 1, -1)
        cv2.rectangle(mask2, (w//2, h//2), (w-10, h-10), 1, -1)
        return [mask1, mask2]

    def detect_furniture(
        image: np.ndarray,
        categories: str,
        confidence: float,
        nms_threshold: float,
        remove_architecture: bool,
        segment_entire_image: bool,
        always_mask_categories: str,
        never_mask_categories: str
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main function. If segment_entire_image is True, we skip YOLO
        and produce all segments from SAM. Otherwise, YOLO+SAM bounding boxes.
        If remove_architecture is True, we run analyze_architectural_elements on the numeric-labeled image.
        
        Returns: logs, annotated_image, unified_overlay, masked_image,
                 number_labeled_image, post_removal_image
        """
        logs = []
        
        # If segment_entire_image is True => skip YOLO
        if segment_entire_image:
            logs.append("[INFO] Segmenting entire image with SAM. Skipping YOLO detection.")
            # We generate all masks from entire_image_segmentation
            all_masks = entire_image_segmentation(image)
            logs.append(f"[INFO] Produced {len(all_masks)} entire-image masks.")
            
            # Build a single "annotated_image" with color-coded masks (like mask annotator).
            # We'll produce a color-coded result ourselves:
            bgr_color = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # color them
            for i,m in enumerate(all_masks):
                color = sv.color_palette(i)
                bgr_color[m.astype(bool)] = color
            annotated_image = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2RGB)
            
            # Build unified overlay
            unified_mask_overlay = image.copy()
            unify = np.zeros(image.shape[:2], dtype=bool)
            for m in all_masks:
                unify |= m
            overlay_mask = np.zeros_like(image)
            overlay_mask[unify] = [255,255,255]
            cv2.addWeighted(overlay_mask, 0.5, unified_mask_overlay, 1.0, 0, unified_mask_overlay)

            # Build masked_image
            masked_image = image.copy()
            masked_image[unify] = [255,255,255]

        else:
            # Normal YOLO approach
            logs.append("[INFO] Using YOLO -> SAM bounding box approach.")
            # YOLO
            furniture_items = [cat.strip() for cat in categories.split(",")]
            yolo_world.set_classes(furniture_items)
            logs.append(f"[INFO] Searching for: {furniture_items}")

            results = yolo_world.infer(image, confidence=confidence)
            detections = sv.Detections.from_inference(results).with_nms(
                class_agnostic=True, threshold=nms_threshold
            )
            logs.append(f"[INFO] Found {len(detections.xyxy)} objects.")
            
            annotated_image = image.copy()
            unified_mask_overlay = image.copy()
            masked_image = image.copy()

            if len(detections.xyxy) > 0:
                # SAM
                sam.set_image(image, image_format="RGB")
                all_masks = []
                for xyxy in detections.xyxy:
                    mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
                    all_masks.append(mask.squeeze())
                logs.append(f"[INFO] Generated {len(all_masks)} segmentation masks with SAM from YOLO boxes.")
                detections.mask = np.array(all_masks)

                # build annotated
                bgr_annot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # use supervision's annotators
                bgr_annot = sv.MaskAnnotator().annotate(bgr_annot, detections)
                bgr_annot = sv.BoundingBoxAnnotator().annotate(bgr_annot, detections)
                furniture_labels = [
                    f"{furniture_items[cid]}: {conf:.2f}"
                    for cid, conf in zip(detections.class_id, detections.confidence)
                ]
                bgr_annot = sv.LabelAnnotator().annotate(bgr_annot, detections, labels=furniture_labels)
                annotated_image = cv2.cvtColor(bgr_annot, cv2.COLOR_BGR2RGB)

                # unify
                unify = np.zeros(image.shape[:2], dtype=bool)
                for m in all_masks:
                    unify |= m
                overlay_mask = np.zeros_like(image)
                overlay_mask[unify] = [255,255,255]
                cv2.addWeighted(overlay_mask, 0.5, unified_mask_overlay, 1.0, 0, unified_mask_overlay)
                masked_image[unify] = [255,255,255]

            else:
                logs.append("[INFO] No furniture from YOLO => no SAM bounding boxes.")
                all_masks = []  # empty

        # 4) Numeric-labeled image
        if not segment_entire_image:
            # If we used YOLO, we might have all_masks or empty
            # If YOLO found no objects, all_masks is empty
            pass
        # else we already have all_masks from entire-image approach

        if (not segment_entire_image) and len(detections.xyxy) == 0:
            # No masks => produce blank
            logs.append("[INFO] No masks => skipping numeric-labeled image.")
            number_labeled_image = np.zeros((1,1,3),dtype=np.uint8)
        else:
            # We have all_masks from either entire image or YOLO approach
            palette = sv.ColorPalette.DEFAULT
            mask_colors = [palette.by_idx(i).as_rgb() for i in range(len(all_masks))]
            if segment_entire_image:
                mask_labels = ["region" for _ in all_masks]
            else:
                mask_labels = [furniture_items[cid] for cid in detections.class_id]
            number_labeled_image = generate_number_overlay(image, all_masks, mask_labels, mask_colors)

        # 5) If remove_architecture => analyze with LLM
        if remove_architecture:
            logs.append("[INFO] Attempting architectural and category removal.")
            if len(all_masks) == 0:
                logs.append("[WARN] No masks => nothing to remove.")
                post_removal_image = np.zeros((1,1,3),dtype=np.uint8)
            else:
                # Sort indices by area to match visualization
                areas = [(i, np.sum(m)) for i, m in enumerate(all_masks)]
                sorted_indices = [idx for idx, _ in sorted(areas, key=lambda x: x[1], reverse=True)]
                logs.append(f"[INFO] Sorted indices: {sorted_indices}")

                # Get architectural numbers plus never-mask categories
                pil_label_img = Image.fromarray(number_labeled_image)
                arch_numbers, reason = analyze_architectural_elements(pil_label_img)
                logs.append(f"[INFO] Architectural numbers from Gemini: {arch_numbers}, reason: {reason}")

                # Now KEEP any numbered mask if it's a carpet/never-mask category
                removed_numbers = set()
                for i, label in enumerate(mask_labels):
                    if i in sorted_indices: 
                        vis_number = sorted_indices.index(i) + 1
                        if any(cat.strip().lower() in label.lower() for cat in never_mask_categories.split(",")):
                            removed_numbers.add(float(vis_number))
                            logs.append(f"[INFO] Removing number {vis_number} (label: {label})")

                # Remove the preserved numbers from architecture numbers
                numbers_to_remove = list(set(arch_numbers) | removed_numbers)
                logs.append(f"[INFO] Original architectural numbers: {arch_numbers}")
                logs.append(f"[INFO] Numbers to remove: {removed_numbers}")
                logs.append(f"[INFO] Final numbers to remove: {numbers_to_remove}")

                # Convert visualization numbers to original indices
                removal_indices = {sorted_indices[int(x) - 1] for x in numbers_to_remove}
                new_masks = [mask for idx, mask in enumerate(all_masks) if idx not in removal_indices]
                logs.append(f"[INFO] After removal, we have {len(new_masks)} masks.")
                
                # rebuild post_removal_image from new_masks
                new_unified = np.zeros(image.shape[:2], dtype=bool)
                for nm in new_masks:
                    new_unified |= nm
                post_removal_image = image.copy()
                new_overlay = np.zeros_like(image)
                new_overlay[new_unified] = [255,255,255]
                cv2.addWeighted(new_overlay, 0.5, post_removal_image, 1.0, 0, post_removal_image)
        else:
            logs.append("[INFO] Architectural removal not requested.")
            post_removal_image = np.zeros((1,1,3),dtype=np.uint8)

        return (
            "\n".join(logs),
            annotated_image,
            unified_mask_overlay,
            masked_image,
            number_labeled_image,
            post_removal_image
        )

    with gr.Blocks() as demo:
        gr.Markdown("""
        # 🏠 Furniture Detection with YOLO-World + EfficientViT SAM
        This demo can:
         1. Run YOLO to detect furniture boxes, then segment with SAM.
         2. Optionally skip YOLO and segment the **entire image** with SAM.
         3. Optionally call Gemini to identify permanent architecture to remove from the mask.
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Room Image")

                categories = gr.Textbox(
                    label="Furniture Categories",
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

                with gr.Row():
                    confidence = gr.Slider(0,1,0.05,step=0.01,label="Detection Confidence")
                    nms_threshold = gr.Slider(0,1,0.99,step=0.01,label="NMS Threshold")

                remove_arch_cb = gr.Checkbox(value=False, label="Remove Architecture w/ Gemini?")
                entire_image_cb = gr.Checkbox(value=False, label="Segment Entire Image w/ SAM?")
                
                run_btn = gr.Button("🔍 Run")

        logs_out = gr.Textbox(label="Logs", lines=10, interactive=False)
        
        with gr.Row():
            out_annotated = gr.Image(label="Detailed Detection")
            out_unified = gr.Image(label="Unified Overlay")
            out_masked = gr.Image(label="Masked Furniture")
        with gr.Row():
            out_lettered = gr.Image(label="Numbered Segments")
            out_removed = gr.Image(label="Post-Removal Image")
        
        # Hook up button
        run_btn.click(
            fn=detect_furniture,
            inputs=[input_image, categories, confidence, nms_threshold, remove_arch_cb, entire_image_cb, always_mask_categories, never_mask_categories],
            outputs=[logs_out, out_annotated, out_unified, out_masked, out_lettered, out_removed],
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
