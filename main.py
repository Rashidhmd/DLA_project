"""
Brain Tumor Segmentation — RF-DETR fine-tuning pipeline.

Usage examples
--------------
Train experiment 1:
    python main.py train --exp 1

Evaluate a checkpoint:
    python main.py test --checkpoint output/fine_tune50_hp_1/checkpoint_best_total.pth

Compare experiments side by side:
    python main.py compare --images 51 65 48 64 47 18 --save comparison.png

Show predictions vs ground truth for specific indices:
    python main.py show --images 51 65 47 64 48 18 --save exp1.jpg
"""

import argparse
import gc
import os
import random
import weakref

import matplotlib.pyplot as plt
import supervision as sv
import torch
from PIL import Image, ImageDraw
from rfdetr import RFDETRSegPreview
from roboflow import Roboflow


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

MODEL_CHECKPOINTS = {
    "Exp 1": "output/fine_tune50_hp_1/checkpoint_best_total.pth",
    "Exp 2": "output/fine_tune50_hp_2/checkpoint_best_total.pth",
    "Exp 3": "output/fine_tune50_hp_3/checkpoint_best_total.pth",
    "Exp 4": "output/fine_tune50_hp_4/checkpoint_best_total.pth",
    "Exp 5": "output/fine_tune50_hp_5/checkpoint_best_total.pth",
}

# Shared fine-tuning defaults that every experiment overrides selectively
_BASE_TRAIN_KWARGS = dict(
    epochs=50,
    batch_size=2,
    grad_accum_steps=8,   # effective batch = 16
    device="cuda",
    gradient_checkpointing=True,
    use_ema=True,
    early_stopping=True,
    early_stopping_patience=15,
    tensorboard=True,
)

EXPERIMENT_CONFIGS = {
    1: dict(
        output_dir="output/fine_tune50_hp_1",
        lr=1e-4, lr_encoder=1.5e-4,
        resolution=432, weight_decay=1e-4,
    ),
    2: dict(
        output_dir="output/fine_tune50_hp_2",
        lr=1e-4, lr_encoder=2e-5,
        resolution=432, weight_decay=1e-4,
    ),
    3: dict(
        output_dir="output/fine_tune50_hp_3",
        lr=1.5e-4, lr_encoder=2e-5,
        resolution=432, weight_decay=1e-4,
    ),
    4: dict(
        output_dir="output/fine_tune50_hp_4",
        lr=1e-4, lr_encoder=2e-5,
        warmup_epochs=1.0, lr_scheduler="cosine", lr_min_factor=0.1,
    ),
    5: dict(
        output_dir="output/fine_tune50_hp_5",
        lr=1e-4, lr_encoder=1e-5,
        warmup_epochs=1.0, lr_scheduler="cosine", lr_min_factor=0.1,
    ),
    # "fast" variants for quick iteration
    "fast": dict(
        output_dir="output/fine_tune50_hp_fast",
        lr=1e-4, lr_encoder=2e-5,
        use_ema=False, num_workers=8, resolution=384,
        multi_scale=False, expanded_scales=False,
        warmup_epochs=1.0, lr_scheduler="cosine", lr_min_factor=0.1,
        early_stopping_patience=10,
    ),
    "fast1": dict(
        output_dir="output/fine_tune50_hp_fast1",
        lr=1e-4, lr_encoder=1e-5,
        use_ema=False, num_workers=8, resolution=384,
        multi_scale=False, expanded_scales=False,
        warmup_epochs=1.0, lr_scheduler="cosine", lr_min_factor=0.1,
        early_stopping_patience=10,
    ),
    "fast2": dict(
        output_dir="output/fine_tune50_hp_fast2",
        lr=1.5e-4, lr_encoder=1e-5,
        use_ema=False, num_workers=8, resolution=384,
        multi_scale=False, expanded_scales=False,
        warmup_epochs=1.0, lr_scheduler="cosine", lr_min_factor=0.1,
        early_stopping_patience=10,
    ),
}


# ---------------------------------------------------------------------------
# GPU / memory helpers
# ---------------------------------------------------------------------------

def cleanup_gpu_memory(obj=None, verbose: bool = False):
    """Free CUDA memory, optionally dropping a model reference first."""
    if not torch.cuda.is_available():
        if verbose:
            print("[INFO] CUDA not available — skipping GPU cleanup.")
        return

    def _stats():
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    torch.cuda.synchronize()
    if verbose:
        a, r = _stats()
        print(f"[Before] Allocated: {a / 1024**2:.2f} MB | Reserved: {r / 1024**2:.2f} MB")

    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            print("[WARNING] Object not fully garbage-collected yet.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    if verbose:
        a, r = _stats()
        print(f"[After]  Allocated: {a / 1024**2:.2f} MB | Reserved: {r / 1024**2:.2f} MB")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def download_dataset():
    """Download the brain-tumor segmentation dataset from Roboflow."""
    api_key = ROBOFLOW_API_KEY
    if not api_key:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY environment variable is not set. "
            "Export it before running: export ROBOFLOW_API_KEY=your_key"
        )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("tutorial-extpj").project("brain-tumor-segmentation-jteuo")
    dataset = project.version(1).download("coco-segmentation")
    return dataset


def load_dataset():
    return download_dataset()


def load_test_split(dataset):
    """Return a supervision DetectionDataset for the test split."""
    return sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/test",
        annotations_path=f"{dataset.location}/test/_annotations.coco.json",
        force_masks=True,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(exp_id, dataset):
    """
    Fine-tune the model for a given experiment id.

    Parameters
    ----------
    exp_id : int | str
        One of the keys in EXPERIMENT_CONFIGS (1-5, 'fast', 'fast1', 'fast2').
    dataset : roboflow Dataset
        The downloaded Roboflow dataset object.
    """
    if exp_id not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment id '{exp_id}'. "
            f"Choose from: {list(EXPERIMENT_CONFIGS.keys())}"
        )

    config = {**_BASE_TRAIN_KWARGS, **EXPERIMENT_CONFIGS[exp_id]}
    print(f"\n{'='*60}")
    print(f"  Starting Experiment {exp_id}")
    print(f"{'='*60}")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    model = RFDETRSegPreview()
    model.train(dataset_dir=dataset.location, **config)
    cleanup_gpu_memory(model, verbose=True)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

_PRED_COLORS = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
])
_GT_COLOR = sv.Color.GREEN
_PRED_POLY_COLOR = sv.Color.RED


def _label_annotator(text_scale, color, offset_y):
    return sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_position=sv.Position.TOP_CENTER,
        text_offset=(0, offset_y),
    )


def _offset_for(labels: list[str]) -> int:
    """Choose vertical label offset based on whether 'notumor' class is present."""
    return 30 if labels and "notumor" in labels[0] else -10


def annotate(image: Image.Image, detections: sv.Detections, classes: dict) -> Image.Image:
    """Annotate with multi-colour polygons and confidence labels (quick preview)."""
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    labels = [
        f"{classes.get(cid, 'unknown')} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]
    out = image.copy()
    out = sv.PolygonAnnotator(color=sv.Color.WHITE).annotate(out, detections)
    out = _label_annotator(text_scale, _PRED_COLORS, _offset_for(labels)).annotate(
        out, detections, labels
    )
    out.thumbnail((1000, 1000))
    return out


def annotate_gt(image: Image.Image, detections: sv.Detections, classes: dict) -> Image.Image:
    """Draw ground-truth polygons in green."""
    color = sv.ColorPalette.from_hex(["#78d378"])
    labels = [classes.get(cid, "unknown") for cid in detections.class_id]
    out = image.copy()
    out = sv.PolygonAnnotator(color=_GT_COLOR).annotate(out, detections)
    out = _label_annotator(1.3, color, _offset_for(labels)).annotate(out, detections, labels)
    out.thumbnail((1000, 1000))
    return out


def annotate_pred(image: Image.Image, detections: sv.Detections, classes: dict) -> Image.Image:
    """Draw predicted polygons in red/orange."""
    color = sv.ColorPalette.from_hex(["#edaa7b"])
    labels = [
        f"{classes.get(cid, 'unknown')} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]
    out = image.copy()
    out = sv.PolygonAnnotator(color=_PRED_POLY_COLOR).annotate(out, detections)
    out = _label_annotator(1.3, color, _offset_for(labels)).annotate(out, detections, labels)
    out.thumbnail((1000, 1000))
    return out


# ---------------------------------------------------------------------------
# Evaluation / visualisation
# ---------------------------------------------------------------------------

def load_models():
    """Load all experiment checkpoints into a dict keyed by experiment name."""
    models = {}
    for name, ckpt in MODEL_CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            print(f"[WARNING] Checkpoint not found, skipping: {ckpt}")
            continue
        m = RFDETRSegPreview(pretrain_weights=ckpt)
        m.optimize_for_inference()
        models[name] = m
    return models


def test_random(checkpoint: str, n: int = 9):
    """
    Predict on N random test images and display them in a 3×3 grid.

    Parameters
    ----------
    checkpoint : str
        Path to the model checkpoint (.pth).
    n : int
        Number of random images to sample (must be a perfect square for a square grid).
    """
    dataset = load_dataset()
    cleanup_gpu_memory(verbose=True)
    model = RFDETRSegPreview(pretrain_weights=checkpoint)
    model.optimize_for_inference()

    ds_test = load_test_split(dataset)
    class_map = {i: name for i, name in enumerate(ds_test.classes)}

    indices = random.sample(range(len(ds_test)), n)
    annotated = []
    for i in indices:
        path, _, _ = ds_test[i]
        image = Image.open(path)
        detections = model.predict(image, threshold=0.5)
        annotated.append(annotate(image, detections, class_map))

    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, img in zip(axes.flat, annotated):
        ax.imshow(img)
        ax.axis("off")
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show()


def show_gt_vs_pred(checkpoint: str, image_indices: list[int], save_path: str = None):
    """
    Display ground-truth (green) beside prediction (red) for selected indices.

    Parameters
    ----------
    checkpoint : str
        Path to the model checkpoint (.pth).
    image_indices : list[int]
        Indices into the test split to visualise. Length must be divisible by 2
        (pairs are shown side-by-side per row).
    save_path : str, optional
        If provided, save the figure to this path.
    """
    dataset = load_dataset()
    cleanup_gpu_memory(verbose=True)
    model = RFDETRSegPreview(pretrain_weights=checkpoint)
    model.optimize_for_inference()

    ds_test = load_test_split(dataset)
    class_map = {i: name for i, name in enumerate(ds_test.classes)}

    n = len(image_indices)
    cols_per_pair = 2           # GT | Pred
    pairs = (n + 1) // 2
    n_rows = pairs
    n_cols = 4                  # two pairs per row

    # If ≤ 3 images, use a simpler layout
    if n <= 3:
        n_rows, n_cols = n, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    for k, idx in enumerate(image_indices):
        image_path, _, gt_detections = ds_test[idx]
        image = Image.open(image_path)

        gt_img = annotate_gt(image, gt_detections, class_map)
        pred_detections = model.predict(image, threshold=0.5)
        pred_img = annotate_pred(image, pred_detections, class_map)

        row = k // 2
        col_base = (k % 2) * 2
        axes[row, col_base].imshow(gt_img)
        axes[row, col_base].set_title("Ground Truth" if k == 0 else "")
        axes[row, col_base].axis("off")

        axes[row, col_base + 1].imshow(pred_img)
        axes[row, col_base + 1].set_title("Prediction" if k == 0 else "")
        axes[row, col_base + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_experiment_grid(
    image_indices: list[int],
    models: dict,
    ds_test,
    class_map: dict,
    save_path: str = None,
):
    """
    Compare all experiments against ground truth for a fixed set of images.

    Layout
    ------
    Rows : Ground Truth + one row per experiment
    Cols : one column per image

    Parameters
    ----------
    image_indices : list[int]
        Exactly 3 test-split indices.
    models : dict
        {experiment_name: model} mapping (from load_models()).
    ds_test : sv.DetectionDataset
    class_map : dict
    save_path : str, optional
    """
    if len(image_indices) != 3:
        raise ValueError("Provide exactly 3 image indices.")

    exp_names = list(models.keys())
    n_rows = 1 + len(exp_names)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    # Row labels in the first column of each row
    row_labels = ["Ground Truth"] + exp_names
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=8)

    for col, img_idx in enumerate(image_indices):
        image_path, _, gt_detections = ds_test[img_idx]
        image = Image.open(image_path)

        axes[0, col].imshow(annotate_gt(image, gt_detections, class_map))
        axes[0, col].axis("off")

        for row, exp_name in enumerate(exp_names, start=1):
            detections = models[exp_name].predict(image, threshold=0.5)
            axes[row, col].imshow(annotate_pred(image, detections, class_map))
            axes[row, col].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved → {save_path}")


def compare_experiments(image_indices: list[int], save_prefix: str = "comparison"):
    """
    Run plot_experiment_grid for two groups of 3 images and save both figures.

    Parameters
    ----------
    image_indices : list[int]
        Exactly 6 test-split indices.
    save_prefix : str
        File name prefix for the two output PNGs.
    """
    if len(image_indices) != 6:
        raise ValueError("Provide exactly 6 image indices.")

    dataset = load_dataset()
    ds_test = load_test_split(dataset)
    class_map = {i: name for i, name in enumerate(ds_test.classes)}
    models = load_models()

    plot_experiment_grid(
        image_indices=image_indices[:3],
        models=models, ds_test=ds_test, class_map=class_map,
        save_path=f"{save_prefix}_images_1_to_3.png",
    )
    plot_experiment_grid(
        image_indices=image_indices[3:],
        models=models, ds_test=ds_test, class_map=class_map,
        save_path=f"{save_prefix}_images_4_to_6.png",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation — RF-DETR fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Fine-tune the model for one or more experiments.")
    t.add_argument(
        "--exp", nargs="+", required=True,
        help="Experiment id(s) to run. E.g. --exp 1 2 3  or  --exp fast",
    )

    # --- test ---
    e = sub.add_parser("test", help="Run predictions on random test images.")
    e.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint.")
    e.add_argument("--n", type=int, default=9, help="Number of random images (default: 9).")

    # --- show ---
    s = sub.add_parser("show", help="Show GT vs prediction for specific indices.")
    s.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint.")
    s.add_argument("--images", nargs="+", type=int, required=True,
                   help="Test-split indices to visualise (even number, pairs per row).")
    s.add_argument("--save", default=None, help="Optional output image path.")

    # --- compare ---
    c = sub.add_parser("compare", help="Compare all experiments on 6 test images.")
    c.add_argument("--images", nargs=6, type=int,
                   default=[51, 65, 48, 64, 47, 18],
                   help="Exactly 6 test-split indices (default: 51 65 48 64 47 18).")
    c.add_argument("--save", default="comparison",
                   help="Output filename prefix (default: 'comparison').")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        dataset = load_dataset()
        for exp_id_str in args.exp:
            # Accept both ints and strings like "fast"
            try:
                exp_id = int(exp_id_str)
            except ValueError:
                exp_id = exp_id_str
            finetune(exp_id, dataset)

    elif args.command == "test":
        test_random(checkpoint=args.checkpoint, n=args.n)

    elif args.command == "show":
        show_gt_vs_pred(
            checkpoint=args.checkpoint,
            image_indices=args.images,
            save_path=args.save,
        )

    elif args.command == "compare":
        compare_experiments(image_indices=args.images, save_prefix=args.save)


if __name__ == "__main__":
    main()
