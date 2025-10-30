# SoM-MIMO: Semantic Segmentation over MIMO Channels

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![Detectron2](https://img.shields.io/badge/Detectron2-Latest-ff69b4.svg)](https://github.com/facebookresearch/detectron2)

## Overview

This is the official implementation of the paper:

**"Synesthesia of Machines (SoM)-Based Task-Driven MIMO System for Image Transmission"**

S. Li, R. Zhang, X. Cheng and J. Tang, in *IEEE Transactions on Wireless Communications*, doi: [10.1109/TWC.2025.3606237](https://doi.org/10.1109/TWC.2025.3606237).

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+ with CUDA support
- Detectron2
- Other dependencies: timm, fvcore, numpy, matplotlib, opencv-python, pycocotools, pillow, iopath, omegaconf, pyyaml, tensorboard

### Setup Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/SoM-MIMO.git
cd SoM-MIMO
```

2. **Install PyTorch** (follow [official instructions](https://pytorch.org/)):


3. **Install Detectron2** (follow [official instructions](https://github.com/facebookresearch/detectron2)):

4. **Install other dependencies**:
```bash
pip install timm fvcore numpy matplotlib opencv-python pycocotools pillow iopath omegaconf pyyaml tensorboard torchviz
```

5. **Prepare Cityscapes Dataset**:
   - Download Cityscapes dataset from [official website](https://www.cityscapes-dataset.com/)
   - Configure the dataset path in `detectron2/detectron2/data/datasets/builtin.py`
   - For detailed instructions, refer to [Detectron2 datasets documentation](https://detectron2.readthedocs.io/tutorials/datasets.html)

## Model Download

Pre-trained model weights are hosted on Hugging Face. Please download them manually:

1. Visit the Hugging Face repository: [https://huggingface.co/sijiangli/SoM-MIMO](https://huggingface.co/sijiangli/SoM-MIMO)
2. Navigate to the "Files and versions" tab
3. Download the following model files:
   - `SoM_MIMO_C12.pth` (Compression rate C=12)
   - `SoM_MIMO_C24.pth` (Compression rate C=24)
   - `SoM_MIMO_C48.pth` (Compression rate C=48)
4. Place all downloaded files in the `model/` directory

The model files should be organized as:
```
model/
├── SoM_MIMO_C12.pth  # Compression rate C=12
├── SoM_MIMO_C24.pth  # Compression rate C=24
└── SoM_MIMO_C48.pth  # Compression rate C=48
```

## Quick Start

After completing the installation and model download steps:

1. **Ensure you have Cityscapes dataset registered** in Detectron2
2. **Run inference** with default settings:
```bash
python inference.py --config-file configs/MIMO/SoM_MIMO_Cityscapes.yaml
```

Results will be saved in the `output/` directory.

## Usage

The inference script supports various configuration options and model variants.

### Using Different Compression Rates

To use different compression rates, modify two files:

1. Update the model path in `configs/MIMO/SoM_MIMO_Cityscapes.yaml`:
```yaml
MODEL:
  WEIGHTS: "model/SoM_MIMO_C12.pth"  # or C24, C48
```

2. Update the compression parameter in `SoM_MIMO/config_city.py`:
```python
cfg.MODEL.MIMO.C = 12  # or 24, or 48
```

## Configuration

Key configuration parameters in `SoM_MIMO/config_city.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `C` | Compression channel dimension | 48 |
| `HFF` | Enable Hierarchical Feature Fusion | 1 |
| `MCE` | Enable MIMO Channel Exploitation | 1 |
| `MCE_DEPTH` | Depth of MCE modules | 6 |
| `BITS` | Quantization bits | 4 |
| `Nt` | Number of transmit antennas | 2 |
| `Nr` | Number of receive antennas | 2 |
| `INFER_SNR` | Inference SNR (dB) | 15 |
| `EVAL_REPEAT` | Number of evaluation repeats | 50 |

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{11159589,
  author={Li, Sijiang and Zhang, Rongqing and Cheng, Xiang and Tang, Jian},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Synesthesia of Machines (SoM)-Based Task-Driven MIMO System for Image Transmission}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={MIMO;Image reconstruction;Precoding;Mobile agents;Image communication;Feature extraction;Instance segmentation;Symbols;Vehicle dynamics;Transformers;SoM;cooperative perception;instance segmentation;MIMO transmission},
  doi={10.1109/TWC.2025.3606237}
}
```

## Acknowledgments

- Built on [Detectron2](https://github.com/facebookresearch/detectron2)
- Uses [timm](https://github.com/rwightman/pytorch-image-models) for vision transformers
- Cityscapes dataset from [official website](https://www.cityscapes-dataset.com/)

---

**Note**: Make sure to download the model weights from [Hugging Face](https://huggingface.co/sijiangli/SoM-MIMO) before running inference.
