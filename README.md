# Model Fingerprinting via Spectral Coupling in Diffusion Models

This repository contains the implementation for **fingerprinting diffusion models** using spectral coupling signatures extracted from the denoising process. Our method enables accurate attribution of generated images to their source models, even distinguishing between closely related variants.

## 🔬 Overview

This work introduces a novel approach to model attribution based on analyzing the **spectral coupling patterns** in diffusion models' denoising trajectories. By probing the model at multiple timesteps with frequency-band-specific perturbations, we extract characteristic signatures that uniquely identify each model.

### Key Features

- **8-Model Attribution**: Distinguishes between 8 different diffusion models across three axes:
  - **Axis 1 (Data)**: SD v1.4, v1.5, v2.1, Dreamshaper-8, Realistic Vision v5
  - **Axis 2 (Architecture)**: SDXL, PixArt-α
  - **Axis 3 (Training Procedure)**: SDXL-Turbo

- **High Accuracy**: Achieves 99.6-99.98% classification accuracy with minimal prototypes (1-50 samples)

- **Cross-Domain Generalization**: Maintains 98.7-99.98% accuracy on MS-COCO test set

- **Trajectory-Based**: Demonstrates that model identity is encoded in the denoising trajectory, not single timesteps

## 📋 Requirements

### System Requirements
- Python 3.8 or higher

### Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `torch>=2.0.0` - PyTorch deep learning framework
- `diffusers>=0.21.0` - Hugging Face diffusion models
- `transformers>=4.30.0` - Text encoders
- `scikit-learn>=1.3.0` - Classification algorithms
- `numpy>=1.24.0`, `scipy>=1.10.0` - Numerical computing
- `datasets>=2.12.0` - Dataset loading (Stable-Diffusion-Prompts)
- `Pillow>=9.5.0` - Image processing
- `tqdm>=4.65.0` - Progress bars

Optional:
- `xformers>=0.0.20` - Memory-efficient attention (recommended for lower VRAM)

## 🚀 Quick Start

### Basic Usage

Run the full 8-model experiment with 1-step and 5-step signatures:

```bash
python eval_all_8_1t_5t.py \
    --proto_n 50 \
    --test_n 500 \
    --end 550 \
    --cuda_visible_devices 0
```

### Parameters

- `--proto_n`: Number of prototype images per model (default: 50)
- `--test_n`: Number of test images per model (default: 500)
- `--end`: Total prompts to process (must be >= proto_n + test_n)
- `--plans`: Comma-separated timestep plans to run (default: "1step,5step")
  - `1step`: Single timestep at T/2
  - `5step`: 5 timesteps spanning 20%-100% of trajectory
- `--cuda_visible_devices`: GPU device ID (default: "2")
- `--image_cache_dir`: Directory for cached generated images (default: "all8_out/image_cache")
- `--sig_cache_dir`: Directory for cached signatures (default: "all8_out/sig_cache_1t5t")
- `--force_resig`: Force recomputation of signatures

### Smoke Test

Quick test with minimal resources:

```bash
python eval_all_8_1t_5t.py \
    --proto_n 5 \
    --test_n 20 \
    --end 25 \
    --cuda_visible_devices 0
```

## 📊 Methodology

### Signature Extraction

Our method extracts signatures through the following process:

1. **Latent Encoding**: Convert images to latent space using VAE
2. **Frequency Decomposition**: Split into 8 radial frequency bands
3. **Targeted Probing**: Inject band-specific noise at multiple timesteps
4. **Residual Analysis**: Measure model's response in each frequency band
5. **Signature Formation**: Concatenate energy distributions across:
   - 3 noise scales
   - T timesteps (1 or 5)
   - 8×8 frequency band interactions
   
Final signature dimension: `3 × T × 64 = 192 (1-step)` or `960 (5-step)`

### Classification Pipeline

1. **Distance-Based (Argmin)**:
   - Compute cosine distances to prototype signatures
   - Classify by nearest prototype

2. **Learned Classifiers**:
   - **PCA-LR**: PCA (64 components) + Logistic Regression
   - **LinearSVC**: Linear SVM on raw signatures
   - **PCA-SVM (RBF)**: PCA + RBF kernel SVM
   - **PCA-kNN**: PCA + k-Nearest Neighbors (k=5)
   - **Random Forest**: 200 trees, no depth limit

All classifiers use 5-fold stratified cross-validation.

## 📁 Output Structure

```
all8_out/
├── image_cache/              # Generated images
│   ├── images_v14/
│   ├── images_v15/
│   ├── ...
│   └── meta_*.csv           # Metadata per model
├── sig_cache_1t5t/          # Cached signatures
│   ├── v14/
│   │   └── <hash>/
│   │       ├── proto_*.npy
│   │       └── test_*.npy
│   ├── ...
│   ├── plan_1step/
│   │   └── summary.json
│   ├── plan_5step/
│   │   └── summary.json
│   └── summary_8model_1t5t_proto_5.json  # Combined results
```

### Output Metrics

Each `summary.json` contains:

- **Distance Metrics**: Top-1, Top-3, Top-5 accuracy, MRR
- **Per-Class Accuracy**: Individual model performance
- **Classifier Ablation**: Cross-validation scores for all classifiers
- **Confusion Matrices**: Detailed misclassification analysis
- **Signature Metadata**: Dimension, timestep count, etc.

## 🎯 Key Results

### In-Domain Performance (Training Set Prompts)

| Prototypes | Plan  | PCA-LR | LinearSVC | Top-1 Argmin |
|-----------|-------|--------|-----------|--------------|
| 1         | 5-step| 98.0%  | 98.5%     | 96.2%        |
| 5         | 5-step| 99.0%  | 99.5%     | 97.8%        |
| 50        | 5-step| 99.7%  | 99.98%    | 99.1%        |

### Cross-Domain Performance (MS-COCO Prompts)

| Prototypes | PCA-LR | LinearSVC |
|-----------|--------|-----------|
| 1         | 98.9%  | 99.98%    |
| 5         | 98.7%  | 99.98%    |
| 50        | 98.7%  | 99.98%    |

### Key Findings

1. **Few-Shot Effective**: Single prototype per model achieves >96% accuracy
2. **Trajectory Matters**: 5-step signatures significantly outperform 1-step
3. **Strong Generalization**: Minimal accuracy drop on out-of-domain prompts
4. **Robust to Fine-tuning**: Successfully distinguishes closely related models (SD v1.4/v1.5/v2.1)

## 🔧 Advanced Usage

### Custom Timestep Plans

Modify `choose_timestep_indices()` to define custom probing strategies:

```python
def choose_timestep_indices(plan: str, T: int) -> List[int]:
    if plan == "custom":
        # Define your timesteps here
        return [10, 20, 30, 40, 49]
    # ... existing plans
```

### Custom Models

Add new models to `ALL_MODELS` dictionary:

```python
ALL_MODELS = {
    "custom_model": {
        "id": "your-org/your-model",
        "type": "sd15",  # or "sdxl", "pixart", "sdxl_turbo"
        "arch": "UNet-860M",
        "axis": 1,
        "gen_size": 512,
        "gen_steps": 50,
        "gen_cfg": 7.5,
        "desc": "Custom Model Name",
    },
    # ... existing models
}
```

### Signature Configuration

Modify `ProbeConfig` to adjust signature extraction:

```python
@dataclass
class ProbeConfig:
    rings: int = 8              # Frequency bands (8 recommended)
    probe_repeats: int = 6      # Repeated probes per timestep
    timestep_plan: str = "5step"
    noise_scales: List[float] = [1.0, 1.25, 1.5]  # Multi-scale probing
    normalize: bool = True      # L1 normalization
    probe_seed: int = 123       # Reproducibility
```

## 📝 Citation

To be updated soon!

<!-- If you use this code in your research, please cite:

```bibtex
@article{yourname2024fingerprinting,
  title={Model Fingerprinting via Spectral Coupling in Diffusion Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
``` -->

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated**: May 2026

