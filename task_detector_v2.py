"""
Key improvements over baseline:
    [ARCH]     Full text sequence fed into decoder memory (not just CLS token)
    [ARCH]     Confidence / objectness head added alongside bbox head
    [ARCH]     Sinusoidal positional embedding initialization (learnable fine-tune)
    [ARCH]     Precomputed task embeddings for zero-cost text "encoding" at inference
    [LOSS]     GIoU weight warmup over first N epochs (avoids early instability)
    [LOSS]     Confidence loss tied to IoU-based soft target
    [LOSS]     Valid-batch counter so avg loss is never diluted by skipped batches
    [TRAIN]    Mixed-precision training via torch.cuda.amp (autocast + GradScaler)
    [TRAIN]    OneCycleLR scheduler with linear warmup + cosine decay
    [TRAIN]    Gradient accumulation (effective large batch without OOM)
    [DATA]     Degenerate box filtering before albumentations
    [DATA]     __getitem__ retry on empty-box samples
    [DATA]     persistent_workers=True on DataLoaders
    [DEPLOY]   Quantization-Aware Training (QAT) enabled from a configurable epoch
    [DEPLOY]   ONNX export with dynamic batch axes
    [DEPLOY]   Quantized int8 model serialization

Memory-leak fixes (vs previous version):
    [FIX-1]    TOKENIZERS_PARALLELISM=false set before any imports so HuggingFace's
               Rust tokenizer never spawns its own thread pool inside forked workers.
    [FIX-2]    All 14 task prompts are pre-tokenized ONCE in the main process at
               dataset construction time.  Workers receive ready-made tensors via
               copy-on-write fork memory — the tokenizer is never called in a worker.
    [FIX-3]    collate_fn is now a plain function with zero closure state; no
               tokenizer reference leaks into worker processes.
    [FIX-4]    PIL Image explicitly deleted after np.array() conversion so the
               decoded pixel buffer is released before albumentations runs.
    [FIX-5]    worker_init_fn sets cv2 and OMP thread counts per-worker (not
               globally), so PyTorch's own CPU parallelism is unaffected.
    [FIX-6]    Text-encoder trainability cached as a bool at __init__ time;
               forward() no longer iterates all parameters on every call.
    [FIX-7]    Bug: conf_target was referenced before assignment in the
               len(targets)==0 branch of compute_loss. Fixed with zeros_like.
    [FIX-8]    Bug: persistent_workers = True traps memory between epochs, set it to False
    [FIX-9]    Remove non_blocking from the list comprehension so CPU tensors are freed immediately
    [FIX-10]   Set pin_memory = False to decrease RAM usage in loader_kwargs
    [FIX-11]   Explicitly Delete tensors instead of waiting for variable overwrite to
               further decrease RAM usage in the train_one_epoch and validate functions.
New Features:
    [FT-1]     Added resume condition to start training from previous checkpoint
"""

# ─────────────────────────────────────────────────────────────
# SECTION 0 ── Environment variables (MUST precede all imports)
# ─────────────────────────────────────────────────────────────
# [FIX-1] Disable HuggingFace tokenizer's internal Rust parallelism.
# When set BEFORE importing transformers, the tokenizer runs single-threaded
# and never spawns the Rust thread pool that accumulates in forked workers.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ctypes
import gc
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm          # auto-detects Jupyter vs terminal
from transformers import DistilBertModel, DistilBertTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────
def _malloc_trim() -> None:
    """
    Tell glibc to release all freed heap memory above the break point back
    to the OS immediately.  This is the only reliable way to reduce the
    process RSS (Resident Set Size) on Linux after a burst of large numpy /
    PIL allocations — Python's allocator and glibc both hold freed memory in
    their arenas by default, making the Kaggle RAM monitor show a permanent
    high-water mark even after all Python objects are freed.

    This is a no-op on non-Linux platforms (macOS, Windows), so it is safe
    to call unconditionally.
    """
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # not Linux / libc not found — silently skip


def _free_memory(force_trim: bool = False) -> None:
    """Run Python's cyclic GC then optionally tell the OS about freed pages."""
    gc.collect()
    if force_trim:
        _malloc_trim()


# ─────────────────────────────────────────────────────────────
# SECTION 1 ── Configuration Dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # ── Paths ────────────────────────────────────────────────
    img_dir: str = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017"
    anno_dir: str = (
        "/kaggle/input/datasets/arjavjain314/"
        "the-coco-tasks-dataset-2/dataset-master/annotations"
    )
    output_dir: str = "./outputs"

    # ── Training ─────────────────────────────────────────────
    num_epochs: int = 15
    # Physical batch size per gradient-accumulation step.
    # Effective batch = batch_size × accumulation_steps  (e.g. 64 × 8 = 512)
    batch_size: int = 64
    accumulation_steps: int = 8
    max_lr: float = 1e-4
    weight_decay: float = 1e-4
    # Fraction of total optimizer steps used for linear LR warm-up
    warmup_pct: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 4

    # ── Model ────────────────────────────────────────────────
    hidden_dim: int = 256
    num_heads: int = 8
    num_decoder_layers: int = 3
    img_size: int = 640

    # ── Loss ─────────────────────────────────────────────────
    # GIoU weight ramps 0 → 2.0 over this many epochs to stabilise early training
    giou_warmup_epochs: int = 3

    # ── Text encoder ─────────────────────────────────────────
    # False  → fine-tune DistilBERT end-to-end
    # True   → freeze it; only visual backbone + decoder train
    freeze_text_encoder: bool = False

    # ── QAT / Export ─────────────────────────────────────────
    use_qat: bool = True
    # # Start QAT after the model has mostly converged on the float task
    qat_start_epoch: int = 10
    export_onnx: bool = True

    # Resume {FT-1}
    resume: str = ""

    # checkpoint path
    ckpt_name: str = "weights/model640after20ep.pth"

    # For using only 1 GPU:
    use_multiple_gpu: bool = True


# ─────────────────────────────────────────────────────────────
# SECTION 2 ── Task Prompts & Shared Tokenizer
# ─────────────────────────────────────────────────────────────
TASK_PROMPTS: Dict[int, str] = {
    1:  "step on something",
    2:  "sit comfortably",
    3:  "place flowers",
    4:  "get potatoes out of fire",
    5:  "water plant",
    6:  "get lemon out of tea",
    7:  "dig hole",
    8:  "open bottle of beer",
    9:  "open parcel",
    10: "serve wine",
    11: "pour sugar",
    12: "smear butter",
    13: "extinguish fire",
    14: "pound carpet",
}

_tokenizer: Optional[DistilBertTokenizer] = None


def get_tokenizer() -> DistilBertTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return _tokenizer


def _precompute_token_tensors() -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenize all 14 task prompts together (with shared padding) in the main
    process and return a lookup dict: { task_id → (input_ids, attention_mask) }.

    Calling this once at dataset construction time means workers NEVER need
    to import or call the tokenizer, eliminating the Rust thread-pool leak.
    All 14 prompts are tokenized in a single batch so padding is consistent
    across every possible batch composition.
    """
    tokenizer = get_tokenizer()
    task_ids  = list(TASK_PROMPTS.keys())
    prompts   = [TASK_PROMPTS[t] for t in task_ids]
    encoded   = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for i, task_id in enumerate(task_ids):
        cache[task_id] = (
            encoded["input_ids"][i],        # [seq_len]  — same len for all 14
            encoded["attention_mask"][i],   # [seq_len]
        )
    return cache


# ─────────────────────────────────────────────────────────────
# DataLoader worker initialiser
# ─────────────────────────────────────────────────────────────
def _worker_init_fn(worker_id: int) -> None:
    """
    Called once per DataLoader worker sub-process at startup.

    Sets OpenCV and OpenMP thread counts to 1 so albumentations does not
    spawn its own thread pools inside already-parallelised worker processes.
    Doing this here (not globally) avoids restricting PyTorch's own CPU
    parallelism in the main process.

    Also re-confirms TOKENIZERS_PARALLELISM=false for belt-and-suspenders
    safety (the env var is inherited from the main process via fork, but
    explicitly setting it here is harmless and makes intent clear).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass  # cv2 not installed — albumentations will use its PIL fallback


# ─────────────────────────────────────────────────────────────
# SECTION 3 ── Albumentations Transforms
# ─────────────────────────────────────────────────────────────
def build_transforms(img_size: int = 640) -> Tuple[A.Compose, A.Compose]:
    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["labels"],
        min_area=10.0,
        min_visibility=0.1,
    )

    train_transforms = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            # FIX: Use ShiftScaleRotate instead of RandomScale.
            # This scales the contents by +/- 10% but keeps the tensor exactly 320x320.
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=0.0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=bbox_params,
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=bbox_params,
    )

    return train_transforms, val_transforms


# ─────────────────────────────────────────────────────────────
# SECTION 4 ── Dataset
# ─────────────────────────────────────────────────────────────
class UnifiedTaskDataset(Dataset):
    """
    Flat dataset that merges all 14 COCO-Tasks annotation files.
    Each sample is one (image, task) pair with ≥1 preferred-object box.

    Memory-leak fixes vs previous version
    --------------------------------------
    • All 14 task prompts tokenized ONCE at __init__ in the main process.
      __getitem__ returns the pre-computed input_ids / attention_mask tensors
      directly, so DataLoader workers never import or call the tokenizer.
    • PIL Image explicitly deleted after np.array() so the decoded pixel
      buffer is released before albumentations allocates its own buffers.
    • Filesystem fallback chain: split dir → train2017 → val2017.
    • Degenerate COCO box filtering before normalisation.
    • __getitem__ retries on empty-box samples (albumentations heavy crop).
    """

    def __init__(
        self,
        split: str = "train",
        img_dir: str = "",
        anno_dir: str = "",
        transforms: Optional[A.Compose] = None,
        max_retries: int = 5,
    ) -> None:
        self.img_dir = os.path.join(img_dir, f"{split}2017")
        self.img_dir_fallbacks: List[str] = [
            os.path.join(img_dir, "train2017"),
            os.path.join(img_dir, "val2017"),
        ]
        self.transforms  = transforms
        self.max_retries = max_retries
        self.samples: List[dict] = []

        # [FIX-2] Pre-tokenize all 14 prompts once here in the main process.
        # Workers receive these tensors via copy-on-write fork memory.
        # The tokenizer is never called inside a worker process.
        self._token_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = (
            _precompute_token_tensors()
        )

        for task_id in range(1, 15):
            json_path = os.path.join(anno_dir, f"task_{task_id}_{split}.json")
            if not os.path.exists(json_path):
                logger.warning(f"Annotation file not found, skipping: {json_path}")
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            # Group category-1 (preferred object) annotations by image_id
            task_targets: Dict[int, List] = {}
            for ann in data["annotations"]:
                if ann["category_id"] == 1:
                    task_targets.setdefault(ann["image_id"], []).append(ann["bbox"])

            for img_id, boxes in task_targets.items():
                self.samples.append(
                    {"image_id": img_id, "task_id": task_id, "boxes": boxes}
                )

        logger.info(f"[{split:5s}] {len(self.samples):,} (image, task) samples across 14 tasks.")

    # ── Helpers ──────────────────────────────────────────────

    def _load_image(self, img_id: int) -> Image.Image:
        fname = f"{img_id:012d}.jpg"
        for directory in [self.img_dir] + self.img_dir_fallbacks:
            path = os.path.join(directory, fname)
            if os.path.exists(path):
                # Fix for high memory usage, use 'with' statement. Use context manager to close the file handle instantly
                with Image.open(path) as img:
                    return img.convert("RGB")
        raise FileNotFoundError(f"Image {fname} not found in any search directory.")

    @staticmethod
    def _coco_to_yolo(
        xmin: float, ymin: float, w: float, h: float, img_w: int, img_h: int
    ) -> Tuple[float, float, float, float]:
        """COCO [xmin, ymin, w, h] → YOLO normalised [cx, cy, w, h]."""
        cx = (xmin + w / 2.0) / img_w
        cy = (ymin + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h
        return cx, cy, nw, nh

    # ── Core item logic ──────────────────────────────────────

    def _process_sample(self, idx: int):
        sample  = self.samples[idx]
        task_id = sample["task_id"]

        # Retrieve pre-computed token tensors (no tokenizer call in worker)
        input_ids, attention_mask = self._token_cache[task_id]

        image    = self._load_image(sample["image_id"])
        img_w, img_h = image.size
        image_np = np.array(image)
        # [FIX-4] Explicitly release the PIL image now that we have the numpy
        # array. Without this, the decoded pixel buffer stays alive alongside
        # image_np until the function returns, doubling peak per-sample RAM.
        del image

        yolo_boxes: List[List[float]] = []
        for (xmin, ymin, w, h) in sample["boxes"]:
            # ── Filter degenerate COCO boxes before normalisation ──
            if w < 1.0 or h < 1.0:
                continue
            cx, cy, nw, nh = self._coco_to_yolo(xmin, ymin, w, h, img_w, img_h)
            # Skip near-zero normalised boxes that would confuse GIoU
            if nw < 0.001 or nh < 0.001:
                continue
            # Clip into valid YOLO range
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            nw = float(np.clip(nw, 0.001, 1.0))
            nh = float(np.clip(nh, 0.001, 1.0))
            yolo_boxes.append([cx, cy, nw, nh])

        labels = [1] * len(yolo_boxes)  # Dummy labels required by albumentations

        if self.transforms:
            transformed  = self.transforms(
                image=image_np, bboxes=yolo_boxes, labels=labels
            )
            # ToTensorV2 uses torch.from_numpy internally — zero-copy, meaning
            # image_tensor shares the underlying numpy buffer with transformed["image"].
            # .clone() makes an independent copy so the numpy buffer (and the full
            # transformed dict) can be freed RIGHT NOW instead of living until
            # collate_fn discards the batch list (32 × ~1.2 MB held simultaneously).
            # image_tensor = transformed["image"].clone()
            image_tensor = transformed["image"]
            final_boxes  = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            # Free the albumentations output dict AND the original full-res array now.
            # Without these dels, image_np (the full-res COCO image, up to 5 MB) stays
            # alive until _process_sample returns, and transformed's numpy arrays stay
            # alive until image_tensor goes out of scope in the DataLoader batch list.
            del transformed, image_np
        else:
            image_tensor = (
                torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )
            del image_np
            final_boxes = torch.tensor(yolo_boxes, dtype=torch.float32)

        return image_tensor, input_ids, attention_mask, final_boxes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Retries on empty boxes (albumentations may drop all boxes on a heavy crop).
        Picks a fresh random index on each retry to avoid infinite looping on
        a single pathological sample.
        """
        for attempt in range(self.max_retries):
            try:
                image_tensor, input_ids, attention_mask, final_boxes = (
                    self._process_sample(idx)
                )
                if len(final_boxes) > 0:
                    return image_tensor, input_ids, attention_mask, final_boxes
            except Exception as exc:
                logger.debug(f"Sample {idx} failed attempt {attempt + 1}: {exc}")
            # Try a different random sample
            idx = np.random.randint(0, len(self.samples))

        # Last-resort fallback: return whatever we got, even if boxes are empty.
        # compute_loss handles the empty-box case gracefully.
        return self._process_sample(idx)


# ─────────────────────────────────────────────────────────────
# SECTION 5 ── Collate Function
# ─────────────────────────────────────────────────────────────
def collate_fn(batch):
    """
    Plain collate function — no closure state, no tokenizer reference.

    [FIX-3] Because __getitem__ now returns pre-tokenized tensors (all
    padded to the same length at dataset construction time), this function
    simply stacks them.  There is nothing to leak into worker processes.
    """
    images         = torch.stack([item[0] for item in batch])
    input_ids      = torch.stack([item[1] for item in batch])
    attention_mask = torch.stack([item[2] for item in batch])
    targets        = [item[3] for item in batch]   # List[Tensor[N_i, 4]]
    return images, input_ids, attention_mask, targets


# ─────────────────────────────────────────────────────────────
# SECTION 6 ── Model
# ─────────────────────────────────────────────────────────────

def _make_sinusoidal(size: int, dim: int) -> torch.Tensor:
    """
    Returns a (size, dim) sinusoidal positional embedding tensor.
    Used as a warm-start for learnable row/col embeddings so the model
    starts with meaningful spatial priors instead of random noise.
    """
    pe  = torch.zeros(size, dim)
    pos = torch.arange(size).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, dim, 2).float() * -(math.log(10_000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class EndToEndTaskDetector(nn.Module):
    """
    Dual-encoder grounding detector.

    Text path  : DistilBERT (frozen) → linear projection
    Vision path: MobileNetV3-Small  → 1×1 conv projection
    Decoder    : TransformerDecoder
                   query  = CLS token embedding  (what to look for)
                   memory = visual tokens + full text tokens  (where to look)
    Heads      : bbox regression (4 values, sigmoid) + objectness (1 value, sigmoid)

    Key design decisions for RISC-V deployment
    -------------------------------------------
    • Text encoder is fully frozen → can be precomputed at export time.
      Call precompute_task_embeddings() to get a lookup table so the
      RISC-V chip never runs DistilBERT.
    • MobileNetV3-Small is the lightest torchvision backbone (~2.5M params).
    • hidden_dim=256 keeps the transformer small.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads:  int = 8,
        num_decoder_layers: int = 3,
        freeze_text_encoder: bool = False,   # True only for Stage 2 / RISC-V
    ) -> None:
        super().__init__()

        # ── Text Encoder (DistilBERT, 66 M params) ───────────────────
        # freeze_text_encoder=False → fine-tune end-to-end (Stage 1, Kaggle)
        # freeze_text_encoder=True  → frozen; precomputed lookup used at RISC-V
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.text_encoder.parameters():
            param.requires_grad = not freeze_text_encoder
        # [FIX-6] Cache as a plain bool so forward() never iterates all params.
        self._text_encoder_trainable: bool = not freeze_text_encoder

        # Project CLS token  → decoder query  [B, 1, hidden_dim]
        self.cls_proj  = nn.Linear(768, hidden_dim)
        # Project full text sequence → extra memory tokens  [B, seq_len, hidden_dim]
        # The decoder cross-attends to both visual AND text memory, so it can
        # relate spatial features to individual words ("pour", "bottle", etc.)
        self.text_proj = nn.Linear(768, hidden_dim)

        # ── Vision Backbone (MobileNetV3-Small, ~1.5 M feature params) ─
        mobilenet = models.mobilenet_v3_small(weights="DEFAULT")
        # features[-1] output: [B, 576, H/32, W/32]  (10×10 for 320px input)
        self.backbone  = mobilenet.features
        self.conv_proj = nn.Conv2d(576, hidden_dim, kernel_size=1)

        # ── 2-D Positional Embeddings (sinusoidal init, learnable) ────
        self.row_embed = nn.Parameter(_make_sinusoidal(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(_make_sinusoidal(50, hidden_dim // 2))

        # ── Transformer Decoder ───────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            batch_first=True, dropout=0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # ── Prediction Heads ──────────────────────────────────────────
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),   # normalised [cx, cy, w, h] ∈ (0, 1)
        )
        # Objectness score: 1 = object present, 0 = background / no detection
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(), # removed Sigmoid due to autocast errors.
        )

    # ── Forward pass ─────────────────────────────────────────

    def forward(
        self,
        images:         torch.Tensor,   # [B, 3, H, W]
        input_ids:      torch.Tensor,   # [B, seq_len]
        attention_mask: torch.Tensor,   # [B, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = images.device

        # ── 1. Text encoding ─────────────────────────────────────────
        # [FIX-6] Use the cached bool (set at __init__) instead of calling
        # any(p.requires_grad for p in self.text_encoder.parameters()) on
        # every forward pass, which iterates ~66 M parameters each time.
        if self._text_encoder_trainable:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out.last_hidden_state   # [B, seq_len, 768]

        # CLS token → the single decoder query (what the model is looking for)
        cls_query   = self.cls_proj(text_features[:, 0, :]).unsqueeze(1)  # [B, 1, D]
        # Full text sequence → injected into cross-attention memory
        text_memory = self.text_proj(text_features)                        # [B, S, D]

        # ── 2. Visual encoding ───────────────────────────────────────
        vis_features = self.backbone(images)        # [B, 576, H', W']
        vis_proj     = self.conv_proj(vis_features) # [B, D,   H', W']
        B, _, H_feat, W_feat = vis_proj.shape

        # Build 2-D sinusoidal positional encoding and add to visual tokens
        pos = torch.cat(
            [
                self.col_embed[:W_feat].unsqueeze(0).repeat(H_feat, 1, 1),
                self.row_embed[:H_feat].unsqueeze(1).repeat(1, W_feat, 1),
            ],
            dim=-1,
        ).flatten(0, 1).unsqueeze(0).to(device)              # [1, H'*W', D]

        vis_memory = vis_proj.flatten(2).permute(0, 2, 1) + pos  # [B, H'*W', D]

        # ── 3. Combined cross-modal memory ────────────────────────────
        # Visual tokens first (dominant signal), then text tokens.
        # The decoder's cross-attention sees both spatial features and
        # word-level semantics in a single unified sequence.
        combined_memory = torch.cat([vis_memory, text_memory], dim=1)  # [B, H'*W'+S, D]

        # ── 4. Grounding decoder ──────────────────────────────────────
        hidden = self.transformer_decoder(
            tgt=cls_query, memory=combined_memory
        )                                          # [B, 1, D]
        hidden = hidden.squeeze(1)                 # [B, D]

        pred_boxes = self.bbox_head(hidden)        # [B, 4]
        pred_conf  = self.conf_head(hidden)        # [B, 1]

        # Apply sigmoid during inference/export so the output is a valid probability (0 to 1).
        # During training, we output raw logits for BCEWithLogitsLoss.
        if not self.training:
            pred_conf = torch.sigmoid(pred_conf)

        return pred_boxes, pred_conf

    # ── Inference utility ─────────────────────────────────────

    @torch.no_grad()
    def precompute_task_embeddings(
        self, device: torch.device
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Precomputes and returns text embeddings for all 14 fixed task prompts.

        Because the text encoder is frozen and the prompts never change,
        we can run DistilBERT exactly once and cache its output.
        At RISC-V inference time, replace the text encoder entirely with
        this lookup table — zero runtime cost for text "encoding".

        Returns
        -------
        cache : dict  { task_id → (cls_query [1,1,D], text_memory [1,S,D]) }
                      Both tensors are on CPU, ready to torch.save().
        """
        tokenizer = get_tokenizer()
        self.eval()
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for task_id, prompt in TASK_PROMPTS.items():
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            text_out   = self.text_encoder(**enc)
            feats      = text_out.last_hidden_state          # [1, S, 768]
            cls_query  = self.cls_proj(feats[:, 0, :]).unsqueeze(1).cpu()
            text_mem   = self.text_proj(feats).cpu()
            cache[task_id] = (cls_query, text_mem)
        logger.info("Precomputed and cached embeddings for all 14 task prompts.")
        return cache


# ─────────────────────────────────────────────────────────────
# SECTION 7 ── Loss Function
# ─────────────────────────────────────────────────────────────

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2]  (in-place safe)."""
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1
    )


def compute_loss(
    pred_boxes:      torch.Tensor,   # [B, 4]
    pred_confs:      torch.Tensor,   # [B, 1]
    target_boxes_list: List[torch.Tensor],  # List of [N_i, 4]
    epoch:           int,
    giou_warmup_epochs: int = 3,
) -> torch.Tensor:
    """
    Per-sample loss with min-cost assignment to nearest GT box.

    GIoU weight warmup
    ------------------
    GIoU returns –1 for completely non-overlapping boxes, so
    `1 – GIoU` = 2.0 early in training before the model converges
    spatially, which can dwarf the L1 signal and destabilise learning.
    We ramp the GIoU contribution from 0 → 2.0 over `giou_warmup_epochs`.

    Confidence loss
    ---------------
    Target confidence = max IoU of the prediction with any GT box.
    This gives a soft, IoU-calibrated signal rather than a hard 0/1 label.
    When no GT boxes exist in a sample, the target confidence is 0.
    """
    giou_weight = 2.0 * min(1.0, (epoch + 1) / max(giou_warmup_epochs, 1))

    total_loss   = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    valid_count  = 0

    for i in range(len(pred_boxes)):
        pred    = pred_boxes[i].unsqueeze(0)   # [1, 4]
        conf    = pred_confs[i]                # [1]
        targets = target_boxes_list[i]         # [N, 4]

        # ── No ground-truth boxes: penalise confidence only ──────────
        if len(targets) == 0:
            # [FIX-7] conf_target was referenced before assignment here in the
            # previous version (it is only defined later in the has-GT branch).
            # Target confidence is 0 when there are no ground-truth boxes.
            conf_loss  = F.binary_cross_entropy_with_logits(
                conf, torch.zeros_like(conf)
            )
            total_loss = total_loss + 0.1 * conf_loss
            continue

        valid_count += 1
        pred_xyxy   = cxcywh_to_xyxy(pred)
        target_xyxy = cxcywh_to_xyxy(targets)

        # ── L1 regression loss ───────────────────────────────────────
        l1_losses = F.l1_loss(
            pred.expand_as(targets), targets, reduction="none"
        ).sum(dim=1)                           # [N]

        # ── GIoU loss ────────────────────────────────────────────────
        giou_matrix = ops.generalized_box_iou(pred_xyxy, target_xyxy)  # [1, N]
        giou_losses = 1.0 - giou_matrix[0]                             # [N]

        combined = 5.0 * l1_losses + giou_weight * giou_losses

        # Match to the single closest ground-truth box
        min_loss = combined.min()

        # ── Soft confidence loss (target = max IoU with any GT) ──────
        with torch.no_grad():
            iou_vals    = ops.box_iou(pred_xyxy, target_xyxy)[0]        # [N]
            conf_target = iou_vals.max().clamp(0.0, 1.0).unsqueeze(0)

        conf_loss  = F.binary_cross_entropy_with_logits(conf, conf_target)
        total_loss = total_loss + min_loss + 0.5 * conf_loss

    # Guard against an all-empty batch (extremely rare with retry logic)
    return total_loss / max(valid_count, 1)


# ─────────────────────────────────────────────────────────────
# SECTION 8 ── Training & Validation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scaler:     GradScaler,
    scheduler:  torch.optim.lr_scheduler._LRScheduler,
    device:     torch.device,
    epoch:      int,
    cfg:        TrainConfig,
    batch_loss_history: List[float],
) -> float:
    model.train()
    total_loss  = 0.0
    valid_count = 0
    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Ep {epoch+1:02d} TRAIN",
        leave=False,          # bar disappears; summary printed below via print()
        dynamic_ncols=True,
    )

    for step, (images, input_ids, attention_mask, target_boxes) in loop:
        # non_blocking=True is only useful with pin_memory=True.
        # Since pin_memory=False, use plain synchronous copies everywhere.
        images         = images.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_boxes   = [b.to(device) for b in target_boxes]

        with autocast():
            pred_boxes, pred_confs = model(images, input_ids, attention_mask)
            loss = compute_loss(
                pred_boxes, pred_confs, target_boxes,
                epoch=epoch, giou_warmup_epochs=cfg.giou_warmup_epochs,
            )
            loss_scaled = loss / cfg.accumulation_steps

        scaler.scale(loss_scaled).backward()

        is_update_step = (
            (step + 1) % cfg.accumulation_steps == 0
            or (step + 1) == len(loader)
        )
        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            # Run Python's cyclic GC on every optimizer step.
            # DataParallel's per-forward-pass module replicas create grad_fn chains
            # that can form reference cycles CPython's refcount misses. gc.collect()
            # finds and breaks them. Calling it every accumulation_steps (not every
            # batch) keeps overhead low (~1-5 ms per call).
            torch.cuda.empty_cache()
            gc.collect()

        loss_val = loss.item()
        # batch_loss_history.append(loss_val)
        if step % 50 == 0:
            batch_loss_history.append(loss_val)

        if loss_val > 0.0:
            total_loss  += loss_val
            valid_count += 1

        loop.set_postfix(
            loss=f"{loss_val:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )
        del images, input_ids, attention_mask, target_boxes
        del pred_boxes, pred_confs, loss, loss_scaled

    return total_loss / max(valid_count, 1)


@torch.no_grad()
def validate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch:  int,
    cfg:    TrainConfig,
) -> float:
    model.eval()
    total_loss  = 0.0
    valid_count = 0

    loop = tqdm(
        loader,
        desc=f"Ep {epoch+1:02d} VAL  ",
        leave=False,
        dynamic_ncols=True,
    )

    for images, input_ids, attention_mask, target_boxes in loop:
        images         = images.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_boxes   = [b.to(device) for b in target_boxes]

        with autocast():
            pred_boxes, pred_confs = model(images, input_ids, attention_mask)
            loss = compute_loss(
                pred_boxes, pred_confs, target_boxes,
                epoch=epoch, giou_warmup_epochs=cfg.giou_warmup_epochs,
            )

        loss_val = loss.item()
        if loss_val > 0.0:
            total_loss  += loss_val
            valid_count += 1

        loop.set_postfix(loss=f"{loss_val:.4f}")
        del images, input_ids, attention_mask, target_boxes
        del pred_boxes, pred_confs, loss

    gc.collect()
    return total_loss / max(valid_count, 1)


def train_and_validate(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler._LRScheduler,
    scaler:       GradScaler,
    device:       torch.device,
    cfg:          TrainConfig,
) -> Tuple[nn.Module, Dict]:
    """
    Returns the trained model AND a history dict:
        {
          "train_epoch": [avg_loss_ep1, avg_loss_ep2, ...],
          "val_epoch":   [avg_loss_ep1, avg_loss_ep2, ...],
          "train_batch": [loss_batch1, loss_batch2, ...],  # every batch
        }
    Pass the history dict to plot_loss_curves() after training.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt     = "weights/model640after20ep.pth"

    history: Dict[str, List[float]] = {
        "train_epoch": [],
        "val_epoch":   [],
        "train_batch": [],
    }

    # Outer epoch bar — stays visible (leave=True) so overall progress is
    # always visible in the Kaggle cell output even as inner bars clear.
    epoch_bar = tqdm(range(cfg.num_epochs), desc="Epochs", leave=True, dynamic_ncols=True)

    for epoch in epoch_bar:

        # ── QAT switchover ─────────────────────────────────────────
        if cfg.use_qat and epoch == cfg.qat_start_epoch:
            print(f"\n[Epoch {epoch+1}] Enabling Quantization-Aware Training (QAT).")
            raw = model.module if isinstance(model, nn.DataParallel) else model
            raw.cpu()
            raw.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
            torch.quantization.prepare_qat(raw, inplace=True)
            raw.to(device)
            if isinstance(model, nn.DataParallel):
                model = nn.DataParallel(raw)
            else:
                model = raw

        # ── Full GC + OS memory release BEFORE each epoch ──────────
        # This returns any heap memory freed during the previous epoch
        # back to the OS before we start allocating for the next one.
        # Without this, glibc holds freed memory in its arena and the
        # Kaggle RAM monitor never shows a decrease between epochs.
        _free_memory(force_trim=True)

        # ── Train ──────────────────────────────────────────────────
        avg_train = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            device, epoch, cfg, history["train_batch"],
        )

        # ── Validate ───────────────────────────────────────────────
        avg_val = validate(model, val_loader, device, epoch, cfg)

        history["train_epoch"].append(avg_train)
        history["val_epoch"].append(avg_val)

        # ── Persistent epoch summary ────────────────────────────────
        # print() writes to Kaggle's permanent cell output; logger.info()
        # goes to stderr which Kaggle may collapse. Use print() here so
        # results survive after inner tqdm bars are cleared.
        summary = (
            f"[Epoch {epoch+1:02d}/{cfg.num_epochs}] "
            f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        tqdm.write(summary)          # writes above the outer epoch bar cleanly
        epoch_bar.set_postfix(train=f"{avg_train:.4f}", val=f"{avg_val:.4f}")

        # ── Save best checkpoint ───────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            raw = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(raw.state_dict(), best_ckpt)
            tqdm.write(f"  ↳ Val loss improved to {best_val_loss:.4f} — saved {best_ckpt}")

        # ── GC + OS memory trim AFTER each epoch ───────────────────
        _free_memory(force_trim=True)

    tqdm.write(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────
# SECTION 8b ── Loss Plotting
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    """
    Save two plots to output_dir:
      • loss_curves_epoch.png  — epoch-averaged train and val loss
      • loss_curves_batch.png  — every-batch train loss (shows intra-epoch trends)

    Both are saved as PNG and also displayed inline in Kaggle notebooks via
    IPython.display (if available).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Epoch-level curves ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = list(range(1, len(history["train_epoch"]) + 1))
    ax.plot(epochs, history["train_epoch"], marker="o", label="Train loss",  color="#2196F3")
    ax.plot(epochs, history["val_epoch"],   marker="s", label="Val loss",    color="#FF5722")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss (per epoch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    epoch_path = os.path.join(output_dir, "loss_curves_epoch.png")
    fig.savefig(epoch_path, dpi=120)
    plt.close(fig)
    print(f"Epoch loss curve saved → {epoch_path}")

    # ── 2. Batch-level curve ──────────────────────────────────
    if history["train_batch"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        batches = list(range(1, len(history["train_batch"]) + 1))
        ax.plot(batches, history["train_batch"], linewidth=0.6,
                color="#2196F3", alpha=0.6, label="Batch loss")

        # Overlay a smoothed trend (running mean over 50 batches)
        window = min(50, len(history["train_batch"]) // 10 or 1)
        if window > 1:
            smoothed = np.convolve(
                history["train_batch"],
                np.ones(window) / window,
                mode="valid",
            )
            ax.plot(
                list(range(window, len(history["train_batch"]) + 1)),
                smoothed,
                linewidth=1.5, color="#FF5722", label=f"Smoothed (w={window})",
            )
        ax.set_xlabel("Batch (global)")
        ax.set_ylabel("Loss")
        ax.set_title("Per-batch Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        batch_path = os.path.join(output_dir, "loss_curves_batch.png")
        fig.savefig(batch_path, dpi=120)
        plt.close(fig)
        print(f"Batch loss curve saved  → {batch_path}")

    # Try to display inline (works in Kaggle / Jupyter)
    try:
        from IPython.display import Image as IPyImage, display
        display(IPyImage(epoch_path))
        if history["train_batch"]:
            display(IPyImage(batch_path))
    except Exception:
        pass   # Not in a notebook — silently skip


# ─────────────────────────────────────────────────────────────
# SECTION 8c ── Inference & Visualisation
# ─────────────────────────────────────────────────────────────

def visualize_prediction(
    model:       nn.Module,
    device:      torch.device,
    cfg:         TrainConfig,
    *,
    img_path:    Optional[str]  = None,
    task_id:     Optional[int]  = None,
    task_prompt: Optional[str]  = None,
    dataset:     Optional["UnifiedTaskDataset"] = None,
    sample_idx:  Optional[int]  = None,
    conf_threshold: float       = 0.3,
    output_path: Optional[str]  = None,
) -> Image.Image:
    """
    Run the model on one image and draw the predicted bounding box.

    Call modes
    ----------
    Mode A — your own image + custom prompt:
        visualize_prediction(model, device, cfg,
                             img_path="my_image.jpg",
                             task_prompt="open bottle of beer")

    Mode B — random sample from a dataset object:
        visualize_prediction(model, device, cfg, dataset=val_dataset)

    Mode C — specific sample from a dataset:
        visualize_prediction(model, device, cfg,
                             dataset=val_dataset, sample_idx=42)

    Parameters
    ----------
    conf_threshold : float
        Box is drawn only when model confidence ≥ this value.
        Lower it (e.g. 0.1) to always draw a box for debugging.
    output_path : str or None
        If given, the annotated image is saved here as PNG/JPEG.

    Returns
    -------
    PIL.Image with the bounding box drawn on it.
    """
    import textwrap

    # ── 1. Determine source ───────────────────────────────────
    if img_path is None and dataset is None:
        raise ValueError("Provide either img_path or dataset.")

    if dataset is not None:
        if sample_idx is None:
            sample_idx = np.random.randint(0, len(dataset))
        raw_sample = dataset.samples[sample_idx]
        actual_task_id = raw_sample["task_id"]
        # Use the dataset's raw image loading (no transforms applied)
        _, val_tfms = build_transforms(cfg.img_size)
        _, img_tfms = build_transforms(cfg.img_size)   # just for Resize+Normalize
        orig_image = dataset._load_image(raw_sample["image_id"])
        gt_boxes_raw = raw_sample["boxes"]          # COCO format for drawing
        if task_id is None:
            task_id = actual_task_id
        if task_prompt is None:
            task_prompt = TASK_PROMPTS[task_id]
        print(f'Sample idx: {sample_idx}  |  Task {task_id}: "{task_prompt}"')
    else:
        orig_image = Image.open(img_path).convert("RGB")
        gt_boxes_raw = []
        if task_id is None and task_prompt is None:
            raise ValueError("Provide task_id or task_prompt when using img_path.")
        if task_prompt is None:
            task_prompt = TASK_PROMPTS[task_id]

    orig_w, orig_h = orig_image.size

    # ── 2. Pre-process for model ──────────────────────────────
    _, val_tfms = build_transforms(cfg.img_size)
    image_np    = np.array(orig_image)
    transformed = val_tfms(image=image_np, bboxes=[], labels=[])
    image_tensor = transformed["image"].clone().unsqueeze(0).to(device)  # [1, 3, H, W]
    del transformed, image_np

    tokenizer   = get_tokenizer()
    encoded     = tokenizer(task_prompt, return_tensors="pt")
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # ── 3. Run model ──────────────────────────────────────────
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    with torch.no_grad():
        pred_box, pred_conf = raw_model(image_tensor, input_ids, attention_mask)

    conf_score  = torch.sigmoid(pred_conf).item() if pred_conf.shape[-1] == 1 else pred_conf.item()
    cx, cy, w, h = pred_box[0].tolist()

    print(f"Predicted box (YOLO norm): cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")
    print(f"Confidence: {conf_score:.3f}  (threshold: {conf_threshold})")

    # ── 4. Convert normalised box → pixel coords (original res) ──
    px_cx = cx * orig_w;  px_cy = cy * orig_h
    px_w  = w  * orig_w;  px_h  = h  * orig_h
    x1 = max(0, int(px_cx - px_w / 2))
    y1 = max(0, int(px_cy - px_h / 2))
    x2 = min(orig_w, int(px_cx + px_w / 2))
    y2 = min(orig_h, int(px_cy + px_h / 2))

    # ── 5. Draw on a copy of the original image ───────────────
    vis = orig_image.copy()
    draw = ImageDraw.Draw(vis)
    box_color = "#00E676" if conf_score >= conf_threshold else "#FF1744"
    line_width = max(2, orig_w // 200)

    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

    label = f"{task_prompt}  conf={conf_score:.2f}"
    # Draw label background
    font_size = max(12, orig_w // 60)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((x1, y1 - font_size - 4), label, font=font)
    draw.rectangle(bbox_text, fill=box_color)
    draw.text((x1, y1 - font_size - 4), label, fill="black", font=font)

    # Draw ground-truth boxes if available (green dashed style via thin rect)
    for (gx, gy, gw, gh) in gt_boxes_raw:
        draw.rectangle(
            [int(gx), int(gy), int(gx + gw), int(gy + gh)],
            outline="#FFEB3B", width=max(1, line_width - 1),
        )

    # ── 6. Display & save ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(np.array(vis))
    ax.axis("off")
    legend_text = (
        f"Task: {task_prompt}\n"
        f"Pred conf: {conf_score:.3f}  ({'ABOVE' if conf_score >= conf_threshold else 'BELOW'} threshold)\n"
        f"Box (px): ({x1},{y1}) → ({x2},{y2})\n"
        + ("GT boxes shown in yellow." if gt_boxes_raw else "")
    )
    ax.set_title(legend_text, fontsize=9, loc="left")
    plt.tight_layout()

    if output_path:
        vis.save(output_path)
        fig.savefig(output_path.replace(".jpg", ".png").replace(".jpeg", ".png"), dpi=120)
        print(f"Saved annotated image → {output_path}")

    try:
        from IPython.display import display as ipy_display
        plt.show()
        ipy_display(vis)
    except Exception:
        plt.show()

    plt.close(fig)
    return vis


# ─────────────────────────────────────────────────────────────
# SECTION 9 ── Export Utilities
# ─────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, cfg: TrainConfig, device: torch.device) -> None:
    """
    Export the floating-point model to ONNX opset 17 with dynamic batch axes.
    Validate the exported graph with onnx.checker if onnx is installed.
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.eval()

    dummy_img  = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
    dummy_ids  = torch.randint(0, 1000, (1, 10), device=device)
    dummy_mask = torch.ones(1, 10, dtype=torch.long, device=device)

    onnx_path = os.path.join(cfg.output_dir, "detector.onnx")

    try:
        torch.onnx.export(
            raw,
            (dummy_img, dummy_ids, dummy_mask),
            onnx_path,
            opset_version=17,
            input_names=["image", "input_ids", "attention_mask"],
            output_names=["pred_boxes", "pred_conf"],
            dynamic_axes={
                "image":          {0: "batch"},
                "input_ids":      {0: "batch"},
                "attention_mask": {0: "batch"},
                "pred_boxes":     {0: "batch"},
                "pred_conf":      {0: "batch"},
            },
        )
        logger.info(f"ONNX model exported → {onnx_path}")

        # Optional validation
        try:
            import onnx
            onnx.checker.check_model(onnx_path)
            logger.info("ONNX graph validation passed.")
        except ImportError:
            logger.info("onnx package not installed; skipping graph validation.")
        except Exception as e:
            logger.warning(f"ONNX graph validation warning: {e}")

    except Exception as e:
        logger.error(f"ONNX export failed: {e}")


def export_quantized(model: nn.Module, cfg: TrainConfig) -> nn.Module:
    """
    Convert a QAT-prepared model to a fully int8 quantised model and save.
    The resulting model can be loaded on RISC-V hardware with QNNPACK support.
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.eval().cpu()

    try:
        quantized_model = torch.quantization.convert(raw, inplace=False)
        out_path = os.path.join(cfg.output_dir, "quantized_detector.pth")
        torch.save(quantized_model.state_dict(), out_path)
        logger.info(f"Quantized int8 model saved → {out_path}")
        return quantized_model
    except Exception as e:
        logger.error(f"Quantized export failed: {e}")
        return raw


# ─────────────────────────────────────────────────────────────
# SECTION 10 ── Argument Parsing
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task-Aware Object Detector — training + export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required paths ────────────────────────────────────────
    paths = p.add_argument_group("Paths (required)")
    paths.add_argument(
        "--img_dir", type=str, required=True,
        help="Root directory of COCO 2017 images (contains train2017/, val2017/)",
    )
    paths.add_argument(
        "--anno_dir", type=str, required=True,
        help="Directory with COCO-Tasks JSON files (task_<N>_train.json, etc.)",
    )
    paths.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory for checkpoints, ONNX, quantized weights, and embedding cache",
    )

    # ── Training ──────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument("--num_epochs",   type=int,   default=15)
    train.add_argument("--batch_size",   type=int,   default=64,
                       help="Physical per-step batch size")
    train.add_argument("--accumulation", type=int,   default=8,
                       help="Gradient accumulation steps (effective batch = batch_size × accumulation)")
    train.add_argument("--max_lr",       type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--warmup_pct",   type=float, default=0.1,
                       help="Fraction of total steps used for LR warm-up (OneCycleLR)")
    train.add_argument("--grad_clip",    type=float, default=1.0)
    train.add_argument("--num_workers",  type=int,   default=4)
    train.add_argument("--giou_warmup",  type=int,   default=3,
                       help="Epochs over which GIoU weight ramps from 0 → 2")
    train.add_argument("--resume", type=str, default="",
                       help="Path to checkpoint to resume from") # [FT-1] add resume arg
    train.add_argument("--ckpt_name",   type=str, default="weights/model640after20ep.pth")
    train.add_argument("--use_multiple_gpu", type=bool, default=True)

    # ── Model ─────────────────────────────────────────────────
    arch = p.add_argument_group("Model architecture")
    arch.add_argument("--hidden_dim",     type=int, default=256)
    arch.add_argument("--num_heads",      type=int, default=8)
    arch.add_argument("--decoder_layers", type=int, default=3)
    arch.add_argument("--img_size",       type=int, default=320)

    # ── Deployment ────────────────────────────────────────────
    deploy = p.add_argument_group("Deployment / export")
    deploy.add_argument(
        "--freeze_text_encoder", action="store_true",
        help="Freeze DistilBERT (Stage 2 / RISC-V mode). Default: fine-tune it (Stage 1).",
    )
    deploy.add_argument("--no_qat",    action="store_true",
                        help="Disable Quantization-Aware Training")
    deploy.add_argument("--qat_start", type=int, default=10,
                        help="Epoch (0-indexed) to enable QAT")
    deploy.add_argument("--no_export", action="store_true",
                        help="Skip ONNX export after training")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# SECTION 11 ── Entry Point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        img_dir            = args.img_dir,
        anno_dir           = args.anno_dir,
        output_dir         = args.output_dir,
        num_epochs         = args.num_epochs,
        batch_size         = args.batch_size,
        accumulation_steps = args.accumulation,
        max_lr             = args.max_lr,
        weight_decay       = args.weight_decay,
        warmup_pct         = args.warmup_pct,
        grad_clip          = args.grad_clip,
        num_workers        = args.num_workers,
        giou_warmup_epochs = args.giou_warmup,
        hidden_dim         = args.hidden_dim,
        num_heads          = args.num_heads,
        num_decoder_layers = args.decoder_layers,
        img_size           = args.img_size,
        use_qat            = not args.no_qat,
        qat_start_epoch    = args.qat_start,
        export_onnx        = not args.no_export,
        freeze_text_encoder= args.freeze_text_encoder,
        resume             = args.resume, # [FT-1] add resume arg
        ckpt_name          = args.ckpt_name,
        use_multiple_gpu   = args.use_multiple_gpu
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    logger.info(f"Device: {device}  |  GPUs available: {n_gpu}")
    logger.info(
        f"Effective batch size: {cfg.batch_size} × {cfg.accumulation_steps} "
        f"= {cfg.batch_size * cfg.accumulation_steps}"
    )

    # ── Transforms ────────────────────────────────────────────
    train_tfms, val_tfms = build_transforms(cfg.img_size)

    # ── Datasets ──────────────────────────────────────────────
    train_dataset = UnifiedTaskDataset("train", cfg.img_dir, cfg.anno_dir, train_tfms)
    val_dataset   = UnifiedTaskDataset("test",  cfg.img_dir, cfg.anno_dir, val_tfms)

    # [FIX-3] collate_fn is now a plain module-level function (Section 5).
    # No build_collate_fn() factory call needed.
    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=False, # [FIX-10] Set this to False to reduce RAM usage
        # [FIX-8] Set persistent_workers = False
        persistent_workers=False,
        # [FIX-5] Sets cv2 and OMP thread counts per-worker without touching
        # the main-process thread counts that PyTorch itself relies on.
        # worker_init_fn=_worker_init_fn, # part of [FIX-10], remove worker_init_fn
        # prefetch_factor = 2 # remove prefetch_factor as part of [FIX-10]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, **loader_kwargs,
    )

    # ── Model ─────────────────────────────────────────────────
    model = EndToEndTaskDetector(
        cfg.hidden_dim, cfg.num_heads, cfg.num_decoder_layers,
        freeze_text_encoder=cfg.freeze_text_encoder,
    )
    model = model.to(device)

    # [FT-1] add resume
    if cfg.resume:
        if os.path.exists(cfg.resume):
            logger.info(f"Resuming weights from {cfg.resume}")
            state_dict = torch.load(cfg.resume, map_location=device)
            # Handle if the saved model was wrapped in DataParallel
            if "module." in list(state_dict.keys())[0]:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            logger.warning(f"Resume checkpoint not found: {cfg.resume}")

    if n_gpu > 1 and cfg.use_multiple_gpu:
        logger.info(f"Wrapping model with DataParallel across {n_gpu} GPUs.")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )

    # ── LR Scheduler (OneCycleLR: warm-up + cosine decay) ─────
    # total_steps = number of times optimizer.step() is called across all epochs.
    # We use ceil so a partial final accumulation window is counted.
    steps_per_epoch = math.ceil(len(train_loader) / cfg.accumulation_steps)
    total_steps     = steps_per_epoch * cfg.num_epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.max_lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_pct,
        anneal_strategy="cos",
    )

    scaler = GradScaler()

    # ── Training ──────────────────────────────────────────────
    model, history = train_and_validate(
        model, train_loader, val_loader,
        optimizer, scheduler, scaler, device, cfg,
    )

    # ── Loss curves ───────────────────────────────────────────
    plot_loss_curves(history, cfg.output_dir)

    # ── Post-training: precompute & cache text embeddings ─────
    # These replace the text encoder at RISC-V inference time.
    raw         = model.module if isinstance(model, nn.DataParallel) else model
    embed_cache = raw.precompute_task_embeddings(device)
    cache_path  = os.path.join(cfg.output_dir, "task_embedding_cache.pt")
    torch.save(embed_cache, cache_path)
    logger.info(f"Task embedding cache saved → {cache_path}")

    # ── ONNX export ───────────────────────────────────────────
    if cfg.export_onnx:
        export_onnx(model, cfg, device)

    # ── Quantized model export ────────────────────────────────
    if cfg.use_qat:
        export_quantized(model, cfg)

    logger.info("All done. Outputs written to: " + cfg.output_dir)


if __name__ == "__main__":
    main()
