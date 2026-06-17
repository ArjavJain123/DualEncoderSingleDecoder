# inference.py
import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from task_detector_v2 import (
    EndToEndTaskDetector, 
    build_transforms, 
    get_tokenizer, 
    TASK_PROMPTS
)
from config import InferenceConfig as cfg


def get_gt_boxes(anno_dir: str, task_id: int, image_id: int) -> list:
    """
    Surgically extracts GT boxes for a specific image without building a full dataset.
    Searches both 'test' and 'train' splits since we don't know where the downloaded image came from.
    """
    gt_boxes = []
    splits = ["test", "train", "val"]
    
    for split in splits:
        json_path = os.path.join(anno_dir, f"task_{task_id}_{split}.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, "r") as f:
            data = json.load(f)
            
        # COCO-Tasks format: category_id == 1 is the preferred object
        for ann in data.get("annotations", []):
            if ann.get("category_id") == 1 and ann.get("image_id") == image_id:
                gt_boxes.append(ann["bbox"]) # [xmin, ymin, w, h] in absolute pixels
                
        # If we found boxes in this split, no need to check the other splits
        if gt_boxes:
            break
            
    return gt_boxes

def run_multi_inference(image_paths: list, task_ids: list):
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Running on: {cfg.device}")

    # 1. Load Model 
    model = EndToEndTaskDetector(
        cfg.hidden_dim, cfg.num_heads, cfg.num_decoder_layers,
        freeze_text_encoder=cfg.freeze_text_encoder
    ).to(cfg.device)

    if not os.path.exists(cfg.weights_path):
        raise FileNotFoundError(f"Weights not found at {cfg.weights_path}")
    
    state_dict = torch.load(cfg.weights_path, map_location=cfg.device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights loaded successfully.\n")

    _, val_tfms = build_transforms(cfg.img_size)
    tokenizer = get_tokenizer()

    # 2. Process Images
    for img_path, task_id in zip(image_paths, task_ids):
        if not os.path.exists(img_path):
            print(f"Skipping {img_path}: File not found.")
            continue

        # Extract integer image_id from filename (e.g., '000000391895.jpg' -> 391895)
        filename = os.path.basename(img_path)
        try:
            image_id = int(filename.split('.')[0])
        except ValueError:
            print(f"Skipping {img_path}: Filename must be a COCO integer ID.")
            continue

        print(f"Processing: {filename} | ID: {image_id} | Task {task_id}: '{TASK_PROMPTS[task_id]}'")
        
        # Fast JSON lookup for Ground Truth
        gt_boxes = get_gt_boxes(cfg.anno_dir, task_id, image_id)
        if not gt_boxes:
            print(f"  ↳ Warning: No Ground Truth found in JSONs for Image {image_id}, Task {task_id}.")

        # Load Raw Image
        orig_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_image.size

        # Preprocess Image & Text
        image_np = np.array(orig_image)
        transformed = val_tfms(image=image_np, bboxes=[], labels=[])
        image_tensor = transformed["image"].unsqueeze(0).to(cfg.device)

        prompt = TASK_PROMPTS[task_id]
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(cfg.device)
        attention_mask = encoded["attention_mask"].to(cfg.device)

        # Forward Pass
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                pred_box, pred_conf = model(image_tensor, input_ids, attention_mask)

        conf_score = torch.sigmoid(pred_conf).item()
        cx, cy, w, h = pred_box[0].tolist()

        # Convert YOLO [0-1] to Pixel Coordinates
        px_cx = cx * orig_w; px_cy = cy * orig_h
        px_w = w * orig_w;   px_h = h * orig_h
        x1 = max(0, int(px_cx - px_w / 2))
        y1 = max(0, int(px_cy - px_h / 2))

        # Plotting
        fig, ax = plt.subplots(1, figsize=(10, 7))
        ax.imshow(orig_image)

        # Draw Prediction (Green)
        color = '#00E676' if conf_score > 0.1 else '#FF1744'
        rect = patches.Rectangle((x1, y1), px_w, px_h, linewidth=3, edgecolor=color, facecolor='none', label=f'Prediction ({conf_score:.2f})')
        ax.add_patch(rect)

        # Draw Ground Truths (Yellow)
        for i, (gx, gy, gw, gh) in enumerate(gt_boxes):
            gt_rect = patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='#FFEB3B', linestyle='--', facecolor='none', label='Ground Truth' if i==0 else "")
            ax.add_patch(gt_rect)

        ax.set_title(f"Task: {prompt.upper()}")
        ax.legend(loc='upper left')
        ax.axis('off')
        
        # Save
        out_file = os.path.join(cfg.output_dir, f"pred_{image_id}_task{task_id}.png")
        plt.savefig(out_file, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  ↳ Saved -> {out_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fast inference with GT overlay.")
    parser.add_argument("--images", nargs='+', required=True, help="List of image file paths (must be named as COCO IDs)")
    parser.add_argument("--tasks", nargs='+', type=int, required=True, help="List of Task IDs")
    args = parser.parse_args()

    if len(args.images) != len(args.tasks):
        raise ValueError("The number of images must exactly match the number of task IDs provided.")
    print("InferenceConfig:")
    for attr, value in vars(cfg).items():
        if not attr.startswith("__") and not callable(value):
            print(f"  {attr}: {value}")
    run_multi_inference(args.images, args.tasks)