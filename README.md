# DFFIR: [Your Paper Title]

[![Python](https://img.shields.io/badge/Python-3.11.5-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Official PyTorch implementation of [Your Paper Title] (Accepted by [Conference/Journal Name])

## 📋 Overview

This repository contains the official implementation of our paper "[Your Paper Title]". 

**Model Parameters:** 28.38 M

## 📊 Results

| Task           | Dataset  | PSNR (dB) | SSIM      |
| -------------- | -------- | --------- | --------- |
| Dehazing       | SOTS     | 29.50     | 0.955     |
| Deraining      | Rain100L | **38.65** | **0.982** |
| Denoise (σ=15) | CBSD68   | 33.98     | **0.945** |
| Denoise (σ=25) | CBSD68   | **31.57** | **0.916** |
| Denoise (σ=50) | CBSD68   | **28.15** | **0.851** |
| **Average**    | -        | **32.10** | **0.929** |

> **Note:** Bold numbers indicate best performance among compared methods.

## 🔧 Requirements

- Python 3.11.5
- PyTorch 2.1.1
- CUDA 11.8 or higher (recommended)

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/DFFIR.git
cd DFFIR
```

### 2. Create conda environment

```bash
conda create -n dffir python=3.11.5
conda activate dffir
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download CLIP model

The code will automatically download the ViT-B/32 CLIP model when first run.

## 📁 Data Preparation

Organize your data as follows:

```
DFFIR/
├── data/
│   └── Train/
│       ├── Denoise/          # Clean images for denoising training
│       ├── Derain/           # Clean images for deraining training
│       └── Dehaze/           # Clean images for dehazing training
└── test/
    ├── denoise/
    │   └── CBSD68/           # CBSD68 test dataset
    ├── derain/
    │   └── Rain100L/         # Rain100L test dataset
    └── dehaze/               # SOTS test dataset
```

## 🎯 Training

Start training with default settings:

```bash
python train_DFFIR.py
```

### Custom training options

```bash
python train_DFFIR.py \
    --batch_size 2 \
    --epochs 500 \
    --lr 1e-4 \
    --patch_size 128 \
    --save_dir ./checkpoints \
    --gpu 0
```

### Training arguments

| Argument        | Default       | Description                    |
| --------------- | ------------- | ------------------------------ |
| `--batch_size`  | 1             | Batch size per GPU             |
| `--epochs`      | 500           | Number of training epochs      |
| `--lr`          | 1e-4          | Learning rate                  |
| `--patch_size`  | 128           | Size of image patches          |
| `--save_dir`    | ./checkpoints | Directory to save checkpoints  |
| `--save_epoch`  | 1             | Save checkpoint every N epochs |
| `--gpu`         | 0             | GPU device ID                  |
| `--num_workers` | 16            | Number of data loading workers |

## 🧪 Testing

Run testing on all benchmarks:

```bash
python test_DFFIR.py
```

The results will be saved to `./checkpoints/final_results/` with per-image PSNR values in filenames.

## 📝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [PromptIR](https://github.com/va1shn9v/PromptIR) for inspiring this work
- [Restormer](https://github.com/swz30/Restormer) for the transformer architecture
- [CLIP](https://github.com/openai/CLIP) for the text encoder

## 📧 Contact

For any questions, please contact: [your-email@example.com]
```

我已经把你的结果填进去了，加粗显示了你方法的最佳表现：
- Deraining: 38.65 / 0.982
- Denoise σ=25: 31.57 / 0.916  
- Denoise σ=50: 28.15 / 0.851
- Average: 32.10 / 0.929

另外 Denoise σ=15 的 SSIM (0.945) 也是最佳的。