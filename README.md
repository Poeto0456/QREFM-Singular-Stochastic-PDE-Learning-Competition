# QREFM — Singular Stochastic PDE Learning Competition

## Overview

This repository contains **Fusion NSPDE**, a hybrid architecture for learning singular stochastic dynamics, specifically the **Φ⁴₂ equation**.

The model combines:
- **DLR Encoder** — physics-aware feature extraction inspired by Regularity Structures
- **Explicit Neural Spectral Solver** — stable drift–diffusion evolution
- **Statistical loss functions** — Log-Spectral + Differentiable ACF ensuring correct temporal memory and roughness

For full mathematical derivations and experiments, see the Technical Report.

---

## Project Structure

```
Competition_Singular_SPDE_learning/
├── configs/
│   └── example.yaml                     # Configuration files
├── evaluation/                          # Metric utilities
├── src/
│   ├── best_fusion_model_trained.pth    # Pretrained weights
│   ├── diffeq_solver.py                 # Explicit neural spectral solver
│   ├── dlr_encoder.py                   # Physics-informed encoder
│   ├── fusion_model.py                  # Full architecture
│   ├── utilities_NSPDE_specacf.py       # [Advanced] Composite Loss (L2 + Spectral + ACF)
│   └── utilities_NSPDE.py               # [Default] Standard Loss (Used for Best Submission)
├── inference.ipynb                      # Reproduce submission (pred.mat)
├── train.ipynb                          # Train model from scratch
├── requirements.txt                     # Dependencies
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Reproduce Submission (Inference)

To generate the official `pred.mat` file:

1. Open `inference.ipynb`
2. Ensure the pretrained weights exist at:
   ```
   src/best_fusion_model_trained.pth
   ```
3. Run all cells
4. Output `pred.mat` will appear in the root directory

### 3. Training (Optional)

1. Open `train.ipynb`
2. Edit `configs/example.yaml` if needed
3. Run all cells
4. New weights will be saved in `src/`

---

## Key Features

### Physics-Aware Encoding
**DLR Encoder** extracts paracontrolled features capable of handling spatial singularities in Φ⁴₂.

### Explicit Spectral Solver
A stable pseudo-spectral integration scheme separating **Drift F** and **Diffusion G** to respect the Itô SPDE structure.

### Statistical Consistency Loss
Composite objective:
- **L2 trajectory loss**
- **Log-spectral loss** (Fourier-domain consistency)
- **Differentiable ACF loss** (temporal autocorrelation preservation)
  
### Note on Loss Functions:
The file utilities_NSPDE.py implements the standard loss used for our final Submission ID 1133, with pretrained weight included Spectral loss. We strictly include utilities_NSPDE_specacf.py as an advanced implementation (L2 + Spectral + ACF) referenced in our Technical Report.
---

## Contact

For questions or collaboration:

**leduyanh1407@gmail.com**
