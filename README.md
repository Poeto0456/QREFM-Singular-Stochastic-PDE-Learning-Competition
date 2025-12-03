# QREFM-Singular-Stochastic-PDE-Learning-Competition

Overview
This repository contains Fusion NSPDE, a hybrid architecture designed to robustly model singular dynamics, specifically the $\Phi^4_2$ SPDE.
Our method integrates:
- Regularity Structures via a DLR Encoder
- Explicit Neural Spectral Solver for stable trajectory simulation
- Statistical losses (Spectral + ACF) to enforce long-range temporal consistency
For mathematical formulation and experiments, refer to our Technical Report.

Project Structure
Competition_Singular_SPDE_learning/
├── configs/                       
│   └── example.yaml               # Hyperparameter configurations
│
├── evaluation/                    
│   └── ...                        # Metrics & loss calculation helpers
│
├── src/                           
│   ├── best_fusion_model_trained.pth   # [CRITICAL] Pre-trained weights
│   ├── diffeq_solver.py                # Explicit Neural Spectral Solver
│   ├── dlr_encoder.py                  # Physics-informed feature extractor
│   ├── fusion_model.py                 # Main hybrid architecture
│   └── utilities_NSPDE.py              # Spectral + ACF loss functions
│
├── inference.ipynb                # [MAIN] Reproduce submission (pred.mat)
├── train.ipynb                    # Train model from scratch
└── requirements.txt               # Python dependencies
