#!/usr/bin/env python3
"""
inference_cli.py
Self-contained interactive CLI for SA-LUT inference.

Supports:
  1. Single image pair - outputs to directory with matched filenames
  2. Batch inference - validates matching filenames with progress bar

Usage:
  python inference_cli.py
  # or
  make inference
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# SA-LUT model imports
from core.module.model import (
    VLog2StyleNet4D,
    QuadrilinearInterpolation_4D,
)

NUM_BASIS = 64  # default number of basis LUTs


# =========================================================================
# Image I/O utilities
# =========================================================================
_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage()


def load_rgb(path: str):
    """Load RGB image as float tensor in [0,1], shape [C,H,W]."""
    img = Image.open(path).convert("RGB")
    return _to_tensor(img)


def save_rgb(t: torch.Tensor, path: str):
    """Save a [C,H,W] float tensor in [0,1] to disk."""
    t = t.clamp(0, 1).cpu()
    _to_pil(t).save(path)


def list_images(folder):
    """List all image files in a folder, sorted by name."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".PNG"}
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1] in exts]
    return sorted(files)


# =========================================================================
# Model loading
# =========================================================================
def load_salut_from_ckpt(ckpt_path: str, device: torch.device):
    """
    Create the SA-LUT model and load weights from checkpoint.
    Expects a .state.pt file alongside the .ckpt file.
    """
    print("[1/4] Initializing model architecture...")
    sys.stdout.flush()

    # Build model (suppress debug prints from model.__init__)
    import io
    import contextlib

    # Temporarily suppress stdout to hide "num_basis this time: 64" debug print
    with contextlib.redirect_stdout(io.StringIO()):
        model = VLog2StyleNet4D(dim=17, num_basis=NUM_BASIS).to(device)

    print(
        f"      ✓ Model initialized ({sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters)"
    )
    sys.stdout.flush()

    # Load checkpoint
    print(f"[2/4] Loading checkpoint: {ckpt_path}")
    sys.stdout.flush()

    # Support both .ckpt and .state.pt paths
    if ckpt_path.endswith(".state.pt"):
        # User provided .state.pt directly
        pure_state_path = ckpt_path
    else:
        # User provided .ckpt, look for corresponding .state.pt
        ckpt_dir, ckpt_base = os.path.split(ckpt_path)
        pure_state_path = os.path.join(ckpt_dir, ckpt_base + ".state.pt")

    if not os.path.exists(pure_state_path):
        raise RuntimeError(
            f"State dict not found: {pure_state_path}\n"
            f"Expected either:\n"
            f"  - <name>.state.pt (direct path), or\n"
            f"  - <name>.ckpt with <name>.ckpt.state.pt alongside it"
        )

    # Check file size for progress estimation
    file_size_mb = os.path.getsize(pure_state_path) / (512 * 512)
    print(f"      Reading {file_size_mb:.1f} MB from disk...")
    sys.stdout.flush()

    start_time = time.time()
    ckpt = {
        "state_dict": torch.load(
            pure_state_path, map_location=device, weights_only=True
        )
    }
    load_time = time.time() - start_time
    print(f"      ✓ Loaded in {load_time:.1f}s")

    state = ckpt.get("state_dict", ckpt)

    print("[3/4] Loading weights into model...")
    sys.stdout.flush()

    # Keep vlog2stylenet.* and strip prefix
    filtered = {
        k.replace("vlog2stylenet.", ""): v
        for k, v in state.items()
        if k.startswith("vlog2stylenet.")
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # Show warnings if any
    if missing:
        print(f"      ⚠ Missing keys: {len(missing)}")
        if len(missing) <= 3:
            for k in missing:
                print(f"        - {k}")
    if unexpected:
        print(f"      ⚠ Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 3:
            for k in unexpected:
                print(f"        - {k}")

    if not missing and not unexpected:
        print(f"      ✓ All weights loaded successfully")

    print("[4/4] Setting model to evaluation mode...")
    sys.stdout.flush()
    model.eval()

    print("\n" + "=" * 60)
    print("✓ Model ready for inference!")
    print("=" * 60)

    return model


# =========================================================================
# Core stylization function
# =========================================================================
@torch.no_grad()
def salut_stylize_one(
    model,
    content_path: str,
    style_path: str,
    out_path: str,
    device: torch.device,
    verbose: bool = True,
):
    """
    Two-stage workflow:
    1) 512×512 inference to get (fused_lut, context_map)
    2) Apply LUT at full resolution via QuadrilinearInterpolation_4D
    """
    if verbose:
        print(f"Stylizing content: {content_path}")
        print(f"Using style: {style_path}")

    # Load full-res inputs (CHW, [0,1])
    content_full = load_rgb(content_path)  # [3,H,W]
    style_full = load_rgb(style_path)  # [3,Hs,Ws]

    H, W = content_full.shape[1], content_full.shape[2]
    if verbose:
        print(f"Content image size: {H}×{W}")

    # Downsample to 512×512 for the network
    content_512 = F.interpolate(
        content_full.unsqueeze(0),
        size=(512, 512),
        mode="bilinear",
        align_corners=True,
    ).to(device)
    style_512 = F.interpolate(
        style_full.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=True
    ).to(device)

    # Forward pass to obtain fused LUT and context map
    if verbose:
        print("Running inference...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    _, fused_lut, context_map = model(style_512, content_512)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if verbose:
        print(
            f"Generated fused LUT: {fused_lut.shape}, context map: {context_map.shape}"
        )

    # Resize context map back to full resolution, concat with full-res content
    context_full = F.interpolate(
        context_map, size=(H, W), mode="bilinear", align_corners=True
    )  # [1,Cctx,H,W]
    content_full_b = content_full.unsqueeze(0).to(device)  # [1,3,H,W]
    combined = torch.cat([context_full, content_full_b], dim=1)  # [1, Cctx+3, H, W]

    # Apply LUT at full resolution
    if verbose:
        print(f"Applying 4D LUT at full resolution ({H}×{W})...")
    quad = QuadrilinearInterpolation_4D()
    if device.type == "cuda":
        torch.cuda.synchronize()
    _, stylized = quad(fused_lut[0].unsqueeze(0), combined)  # [1,3,H,W] in [0,1]
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Save
    (
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.dirname(out_path)
        else None
    )
    save_rgb(stylized.squeeze(0), out_path)
    if verbose:
        print(f"[ok] wrote stylized image → {out_path}")


# =========================================================================
# Batch validation
# =========================================================================
def validate_batch_directories(content_dir, style_dir):
    """
    Validate that:
    1. Both directories exist
    2. Both contain the same number of images
    3. Image filenames match between directories

    Returns:
        tuple: (is_valid: bool, message: str, matched_pairs: list)
    """
    if not os.path.isdir(content_dir):
        return False, f"Content directory does not exist: {content_dir}", []

    if not os.path.isdir(style_dir):
        return False, f"Style directory does not exist: {style_dir}", []

    content_files = list_images(content_dir)
    style_files = list_images(style_dir)

    if not content_files:
        return False, f"No images found in content directory: {content_dir}", []

    if not style_files:
        return False, f"No images found in style directory: {style_dir}", []

    if len(content_files) != len(style_files):
        return (
            False,
            f"Mismatch: {len(content_files)} content images vs {len(style_files)} style images",
            [],
        )

    # Check that filenames match
    content_set = set(content_files)
    style_set = set(style_files)

    missing_in_style = content_set - style_set
    missing_in_content = style_set - content_set

    if missing_in_style or missing_in_content:
        error_msg = "Filename mismatch detected:\n"
        if missing_in_style:
            error_msg += (
                f"  - In content but not in style: {sorted(missing_in_style)[:5]}"
            )
            if len(missing_in_style) > 5:
                error_msg += f" ... and {len(missing_in_style) - 5} more"
            error_msg += "\n"
        if missing_in_content:
            error_msg += (
                f"  - In style but not in content: {sorted(missing_in_content)[:5]}"
            )
            if len(missing_in_content) > 5:
                error_msg += f" ... and {len(missing_in_content) - 5} more"
        return False, error_msg, []

    # All filenames match
    matched_pairs = [
        (os.path.join(content_dir, fname), os.path.join(style_dir, fname), fname)
        for fname in content_files
    ]

    return (
        True,
        f"Validation passed: {len(matched_pairs)} matching image pairs found",
        matched_pairs,
    )


# =========================================================================
# User interaction utilities
# =========================================================================
def prompt_user_choice():
    """Prompt user to select inference mode."""
    print("\n" + "=" * 60)
    print("SA-LUT Inference CLI")
    print("=" * 60)
    print("\nSelect inference mode:")
    print("  1. Single image pair")
    print("  2. Batch inference")
    print()

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")


def prompt_path(prompt_text, must_exist=True, is_dir=False, default=None):
    """Prompt user for a file/directory path with validation."""
    while True:
        if default:
            path = input(f"{prompt_text} (default: {default}): ").strip()
            if not path:
                path = default
        else:
            path = input(f"{prompt_text}: ").strip()
            if not path:
                print("Path cannot be empty. Please try again.")
                continue

        path = os.path.expanduser(path)

        if must_exist:
            if is_dir:
                if not os.path.isdir(path):
                    print(f"Directory does not exist: {path}")
                    continue
            else:
                if not os.path.isfile(path):
                    print(f"File does not exist: {path}")
                    continue

        return path


def find_available_checkpoints(ckpt_dir="ckpts/salut_ckpt"):
    """Find all available .state.pt checkpoint files."""
    if not os.path.exists(ckpt_dir):
        return []

    checkpoints = []
    for file in os.listdir(ckpt_dir):
        if file.endswith(".state.pt"):
            full_path = os.path.join(ckpt_dir, file)
            # Extract epoch number for sorting
            import re

            match = re.search(r"epoch=(\d+)", file)
            epoch = int(match.group(1)) if match else 0

            checkpoints.append({"path": full_path, "name": file, "epoch": epoch})

    # Sort by epoch number
    checkpoints.sort(key=lambda x: x["epoch"])
    return checkpoints


def prompt_checkpoint_selection(checkpoints):
    """Prompt user to select a checkpoint from available options."""
    print("\n" + "-" * 60)
    print("Available Checkpoints")
    print("-" * 60)

    for i, ckpt in enumerate(checkpoints, 1):
        size_mb = os.path.getsize(ckpt["path"]) / (512 * 512)
        print(f"  {i}. {ckpt['name']}")
        print(f"     Epoch: {ckpt['epoch']}, Size: {size_mb:.1f} MB")

    print()

    while True:
        choice = input(
            f"Select checkpoint (1-{len(checkpoints)}, default: {len(checkpoints)}): "
        ).strip()

        # Default to last (highest epoch) checkpoint
        if not choice:
            return checkpoints[-1]["path"]

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]["path"]
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(checkpoints)}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")


def generate_output_filename(content_path: str, style_path: str) -> str:
    """
    Generate output filename based on content and style filenames.
    Format: content_<content_name>_style_<style_name>.<ext>
    """
    content_name = Path(content_path).stem
    style_name = Path(style_path).stem
    content_ext = Path(content_path).suffix

    return f"content_{content_name}_style_{style_name}{content_ext}"


# =========================================================================
# Inference modes
# =========================================================================
def run_single_inference(model, device):
    """Interactive single image pair inference."""
    print("\n" + "-" * 60)
    print("Single Image Pair Inference")
    print("-" * 60)

    content_path = prompt_path(
        "Enter content image path", must_exist=True, is_dir=False
    )
    style_path = prompt_path("Enter style image path", must_exist=True, is_dir=False)
    output_dir = prompt_path(
        "Enter output directory", must_exist=False, is_dir=True, default="outputs"
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    output_filename = generate_output_filename(content_path, style_path)
    output_path = os.path.join(output_dir, output_filename)

    print("\n" + "-" * 60)
    print("Starting stylization...")
    print(f"Output will be saved as: {output_filename}")
    print("-" * 60)

    salut_stylize_one(model, content_path, style_path, output_path, device)

    print("\n" + "=" * 60)
    print("Stylization complete!")
    print(f"Output saved to: {output_path}")
    print("=" * 60)


def run_batch_inference(model, device):
    """Interactive batch inference with validation."""
    print("\n" + "-" * 60)
    print("Batch Inference")
    print("-" * 60)

    content_dir = prompt_path(
        "Enter content images directory", must_exist=True, is_dir=True
    )
    style_dir = prompt_path(
        "Enter style images directory", must_exist=True, is_dir=True
    )

    # Validate directories
    print("\nValidating directories...")
    is_valid, message, matched_pairs = validate_batch_directories(
        content_dir, style_dir
    )

    print(message)

    if not is_valid:
        print("\nBatch inference aborted due to validation errors.")
        return

    output_dir = prompt_path(
        "Enter output directory", must_exist=False, is_dir=True, default="outputs"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 60)
    print(f"Processing {len(matched_pairs)} image pairs...")
    print("-" * 60)

    # Process with progress bar
    for content_path, style_path, filename in tqdm(
        matched_pairs, desc="Stylizing", unit="pair"
    ):
        output_path = os.path.join(output_dir, filename)
        salut_stylize_one(
            model, content_path, style_path, output_path, device, verbose=False
        )

    print("\n" + "=" * 60)
    print("Batch stylization complete!")
    print(f"All {len(matched_pairs)} images saved to: {output_dir}")
    print("=" * 60)


# =========================================================================
# Main entry point
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SA-LUT Interactive Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script provides an interactive CLI for SA-LUT inference.
No command-line arguments are required - all inputs are prompted interactively.

Optional arguments:
  --ckpt PATH    Override checkpoint path (default: auto-detect)
  --cpu          Force CPU inference (default: use CUDA if available)
""",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint (optional - will search if not provided)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()

    # Early feedback - before any slow operations
    print("\n" + "=" * 60)
    print("SA-LUT Inference - Initialization")
    print("=" * 60)
    print("\nInitializing PyTorch and CUDA...")
    sys.stdout.flush()

    # Setup device
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"         GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"         Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Find or use checkpoint
    if args.ckpt is None:
        # Auto-detect checkpoints in default location
        checkpoints = find_available_checkpoints("ckpts/salut_ckpt")

        if len(checkpoints) == 0:
            print("\n[Checkpoint] No checkpoints found in ckpts/salut_ckpt/")
            ckpt_path = prompt_path(
                "Enter checkpoint path (.state.pt or .ckpt)",
                must_exist=True,
                is_dir=False,
            )
        elif len(checkpoints) == 1:
            # Only one checkpoint, use it automatically
            ckpt_path = checkpoints[0]["path"]
            print(f"\n[Checkpoint] Auto-selected: {checkpoints[0]['name']}")
        else:
            # Multiple checkpoints, let user choose
            ckpt_path = prompt_checkpoint_selection(checkpoints)
            selected_name = os.path.basename(ckpt_path)
            print(f"\n[Checkpoint] Selected: {selected_name}")
    else:
        ckpt_path = args.ckpt
        print(f"\n[Checkpoint] {ckpt_path}")

    # Load model
    print("\n" + "-" * 60)
    print("Loading SA-LUT Model")
    print("-" * 60 + "\n")
    model = load_salut_from_ckpt(ckpt_path, device)

    # Main interaction loop
    choice = prompt_user_choice()

    if choice == 1:
        run_single_inference(model, device)
    elif choice == 2:
        run_batch_inference(model, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInference cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
