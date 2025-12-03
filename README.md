# QREFM — Singular Stochastic PDE Learning Competition

## Overview

This repository contains a hybrid architecture for learning singular stochastic dynamics.

The model combines:
- **DLR Encoder** — feature extraction inspired by Regularity Structures
- **Neural SPDE Solver** — stable drift–diffusion evolution
- **Composite loss functions** — Log-Spectral + Differentiable ACF ensuring correct temporal memory and roughness

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
│   ├── utilities_NSPDE_specaf.py        # Spectral + ACF losses
│   └── utilities_NSPDE.py               # Standard loss
├── inference.ipynb                      # Reproduce submission (pred.mat)
├── train.ipynb                          # Train model from scratch
├── requirements.txt                     # Dependencies
└── README.md                            # This file
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

### DLR Encoder
Extracts features capable of handling singularities.

### Neural SPDE Solver
A stable pseudo-spectral integration scheme separating **Drift F** and **Diffusion G**.

### Composite Loss
Composite objective:
- **L2 loss**
- **Log-spectral loss** 
- **Differentiable ACF loss** 

**Note on Loss Functions:**  
The file `utilities_NSPDE.py` implements the standard loss used for our final Submission ID 1133, with pretrained weight included Spectral loss. We strictly include `utilities_NSPDE_specacf.py` as an advanced implementation (L2 + Spectral + ACF) referenced in our Technical Report.
