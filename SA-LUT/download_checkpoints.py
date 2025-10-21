#!/usr/bin/env python3
"""
download_checkpoints.py
Download SA-LUT model checkpoints from HuggingFace Hub.

Usage:
  python download_checkpoints.py
  # or
  make download-ckpts
"""

import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Please install it: pip install huggingface-hub")
    sys.exit(1)


# Configuration
REPO_ID = "zrgong/SA-LUT"
CHECKPOINT_DIR = "ckpts/salut_ckpt"

# Checkpoint to download
CHECKPOINT_FILE = "epoch=100-step=4127466.ckpt.state.pt"


def download_checkpoint(repo_id, filename, local_dir):
    """Download a single checkpoint file from HuggingFace Hub."""
    print(f"\n[{filename}]")
    print(f"  Downloading from {repo_id}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            repo_type="model"
        )

        # Get file size
        size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
        print(f"  ✓ Downloaded: {size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print("="*60)
    print("SA-LUT Checkpoint Downloader")
    print("="*60)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"\nCheckpoint directory: {CHECKPOINT_DIR}")

    # Check if checkpoint already exists
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\n✓ Checkpoint already exists:")
        print(f"  {CHECKPOINT_FILE} ({size_mb:.1f} MB)")
        print("\n" + "="*60)
        print("No download needed!")
        print("="*60)
        return

    print(f"\nNeed to download:")
    print(f"  - {CHECKPOINT_FILE} (~208 MB)")

    # Confirm download
    print()
    response = input("Download checkpoint? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Download cancelled.")
        return

    # Download checkpoint
    print("\n" + "-"*60)
    print("Downloading checkpoint...")
    print("-"*60)

    success = download_checkpoint(REPO_ID, CHECKPOINT_FILE, CHECKPOINT_DIR)

    # Summary
    print("\n" + "="*60)
    if success:
        print("✓ Checkpoint downloaded successfully!")
        print("="*60)
        print(f"\nCheckpoint location: {os.path.abspath(checkpoint_path)}")
        print(f"You can now run: make inference")
    else:
        print("✗ Download failed!")
        print("="*60)
        print("\nPlease try:")
        print(f"  1. Check your internet connection")
        print(f"  2. Visit https://huggingface.co/{REPO_ID}")
        print(f"  3. Manually download to: {CHECKPOINT_DIR}/")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
