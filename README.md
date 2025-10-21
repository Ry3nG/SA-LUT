# SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer

🎉 **Accepted at ICCV 2025**

Official PyTorch implementation of **SA-LUT**.

**Paper**: [SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer](https://arxiv.org/abs/2506.13465)
📄 [arXiv](https://arxiv.org/abs/2506.13465) | 🤗 [PST50 Dataset](https://huggingface.co/datasets/zrgong/PST50) | 🤗 [Model Checkpoint](https://huggingface.co/zrgong/SA-LUT) | 📚 [BibTeX](#citation)

![Model Pipeline](assets/pipeline.png)

---

## Abstract

Photorealistic style transfer (PST) enables real-world color grading by adapting reference image colors while preserving content structure. Existing methods mainly follow either approaches: generation-based methods that prioritize stylistic fidelity at the cost of content integrity and efficiency, or global color transformation methods such as LUT, which preserve structure but lack local adaptability. To bridge this gap, we propose Spatial Adaptive 4D Look-Up Table (SA-LUT), combining LUT efficiency with neural network adaptability. SA-LUT features: (1) a Style-guided 4D LUT Generator that extracts multi-scale features from the style image to predict a 4D LUT, and (2) a Context Generator using content-style cross-attention to produce a context map. This context map enables spatially-adaptive adjustments, allowing our 4D LUT to apply precise color transformations while preserving structural integrity. To establish a rigorous evaluation framework for photorealistic style transfer, we introduce PST50, the first benchmark specifically designed for PST assessment. Experiments demonstrate that SA-LUT substantially outperforms state-of-the-art methods, achieving a 66.7% reduction in LPIPS score compared to 3D LUT approaches, while maintaining real-time performance at 16 FPS for video stylization.

---

## Overview

SA-LUT is a novel approach for photorealistic style transfer that combines:
- **4D Look-Up Tables (LUTs)** for efficient color and texture mapping
- **Spatial adaptation** for content-aware stylization
- **Two-stage processing** (network inference at low-res + LUT application at full-res)

This enables high-quality photorealistic style transfer at any resolution while maintaining efficiency.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Ry3nG/SA-LUT.git
cd SA-LUT

# 2. Set up environment (creates conda env, installs dependencies & CUDA extensions)
make setup

# 3. Activate environment
conda activate salut_env

# 4. Download model checkpoint (~208 MB from HuggingFace)
make download-ckpts

# 5. Run inference (interactive CLI)
make inference
```

---

## Installation

### Prerequisites

- Linux system (tested on Ubuntu/CentOS)
- CUDA-capable GPU (recommended) or CPU
- Conda package manager
- CUDA toolkit 11.x or 12.x (for GPU support)

### Setup Steps

**1. Create Environment**

```bash
make setup
```

This will:
- Create a conda environment named `salut_env`
- Install all Python dependencies from `environment.yml`
- Build and install custom CUDA extensions (`quadrilinear_cpp`, `trilinear_cpp`)

**2. Activate Environment**

```bash
conda activate salut_env
```

**3. Download Model Checkpoint**

```bash
make download-ckpts
```

Alternatively, manually download from [https://huggingface.co/zrgong/SA-LUT](https://huggingface.co/zrgong/SA-LUT) and place in `ckpts/salut_ckpt/`.

---

## Usage

### Interactive CLI (Recommended)

```bash
make inference
```

The CLI offers two modes:

**Mode 1: Single Image Pair**
- Stylize one content image with one style image
- Prompts for:
  - Content image path
  - Style image path
  - Output directory (default: `outputs/`)
- Output automatically named: `content_<name>_style_<name>.<ext>`

**Mode 2: Batch Inference**
- Process multiple image pairs
- Prompts for:
  - Content images directory
  - Style images directory
  - Output directory (default: `outputs/`)
- **Requirements**:
  - Same number of images in both directories
  - **Matching filenames** between content and style directories
- Shows progress bar during processing

### Command Line

**Single image pair:**
```bash
python inference_cli.py \
  --ckpt ckpts/salut_ckpt/epoch=100-step=4127466.ckpt.state.pt
# Then follow interactive prompts
```

**Force CPU mode:**
```bash
python inference_cli.py --cpu
```

**Custom checkpoint:**
```bash
python inference_cli.py --ckpt /path/to/custom.ckpt.state.pt
```

---

## Example Workflow

### Single Image Stylization

```bash
$ make inference

============================================================
SA-LUT Inference CLI
============================================================

Select inference mode:
  1. Single image pair
  2. Batch inference

Enter your choice (1 or 2): 1

Enter content image path: data/PST50/content_709/1.png
Enter style image path: data/PST50/paired_style/1.png
Enter output directory (default: outputs):

Starting stylization...
Output will be saved as: content_1_style_1.png
------------------------------------------------------------
Stylizing content: data/PST50/content_709/1.png
Using style: data/PST50/paired_style/1.png
Content image size: 1920×1080
...

Stylization complete!
Output saved to: outputs/content_1_style_1.png
```

### Batch Processing

For batch processing, ensure your files are organized with matching names:

```
my_content/
  ├── photo1.jpg
  ├── photo2.jpg
  └── photo3.jpg

my_styles/
  ├── photo1.jpg   # Matches content/photo1.jpg
  ├── photo2.jpg   # Matches content/photo2.jpg
  └── photo3.jpg   # Matches content/photo3.jpg
```

Then run:

```bash
$ make inference

Select inference mode:
  1. Single image pair
  2. Batch inference

Enter your choice (1 or 2): 2

Enter content images directory: my_content
Enter style images directory: my_styles

Validating directories...
Validation passed: 3 matching image pairs found

Enter output directory (default: outputs): my_results

Processing 3 image pairs...
Stylizing: 100%|████████████████| 3/3 [00:15<00:00,  5.2s/pair]

Batch stylization complete!
All 3 images saved to: my_results
```

---

## PST50 Benchmark Dataset

The PST50 dataset is the first benchmark for photorealistic style transfer evaluation.

### Download Dataset

```bash
python << EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='zrgong/PST50',
    repo_type='dataset',
    local_dir='data/PST50'
)
EOF
```

### Dataset Structure

```
data/PST50/
├── content_709/      # 50 content images (Rec.709 color space)
├── content_log/      # 50 content images (log color space)
├── paired_gt/        # 50 ground truth stylizations
├── paired_style/     # 50 style references (paired evaluation)
├── unpaired_style/   # 51 style references (unpaired evaluation)
└── video/            # Video sequences for temporal consistency testing
```

All images are numbered `1.png` to `50.png` for easy pairing.

### Evaluation

- **Paired**: Compare outputs against `paired_gt/` using LPIPS, PSNR, SSIM, H-Corr
- **Unpaired**: Use `unpaired_style/` for qualitative assessment
- **Video**: Test temporal consistency with video sequences

---

## Project Structure

```
SA-LUT/
├── core/                      # Model implementation
│   ├── module/
│   │   ├── model.py          # SA-LUT architecture
│   │   ├── clut4d.py         # 4D LUT operations
│   │   └── interpolation.py  # Interpolation layers
│   └── dataset/
├── ckpts/
│   ├── vgg_normalised.pth    # VGG encoder weights
│   └── salut_ckpt/
│       └── epoch=100-step=4127466.ckpt.state.pt  # Main checkpoint
├── data/
│   └── PST50/                # Evaluation dataset (download separately)
├── quadrilinear_cpp/         # Custom CUDA extension for 4D interpolation
├── trilinear_cpp_torch1.11/  # Custom CUDA extension for 3D interpolation
├── inference_cli.py          # Main inference script (interactive)
├── download_checkpoints.py   # Checkpoint downloader
├── Makefile                  # Convenience commands
├── environment.yml           # Conda dependencies
└── README.md                 # This file
```

---

## Citation

If you use SA-LUT in your research, please cite:

```bibtex
@misc{gong2025salutspatialadaptive4d,
      title={SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer},
      author={Zerui Gong and Zhonghua Wu and Qingyi Tao and Qinyue Li and Chen Change Loy},
      year={2025},
      eprint={2506.13465},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.13465},
}
```

---
