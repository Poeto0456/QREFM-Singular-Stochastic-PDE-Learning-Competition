# QREFM — Singular Stochastic PDE Learning Competition

Overview

This repository contains Fusion NSPDE, a hybrid architecture for learning singular stochastic dynamics, specifically the Φ⁴₂ equation.
The model combines:

DLR Encoder (physics-aware feature extraction inspired by Regularity Structures)

Explicit Neural Spectral Solver for stable drift–diffusion evolution

Statistical loss functions (Log-Spectral + Differentiable ACF) ensuring correct temporal memory and roughness

For full mathematical derivations and experiments, see the Technical Report.

Project Structure

Competition_Singular_SPDE_learning/
• configs/ — config files (example.yaml)
• evaluation/ — metric utilities
• src/
 • best_fusion_model_trained.pth — pretrained weights
 • diffeq_solver.py — explicit neural spectral solver
 • dlr_encoder.py — physics-informed encoder
 • fusion_model.py — full architecture
 • utilities_NSPDE.py — spectral + ACF losses
• inference.ipynb — reproduce submission (pred.mat)
• train.ipynb — train model from scratch
• requirements.txt — dependencies

Quick Start
1. Installation

Requires Python 3.8+
pip install -r requirements.txt

2. Reproduce Submission (Inference)

To generate the official pred.mat file:

Open inference.ipynb

Ensure the pretrained weights exist at:
src/best_fusion_model_trained.pth

Run all cells

Output pred.mat will appear in the root directory

3. Training (Optional)

Open train.ipynb

Edit configs/example.yaml if needed

Run all cells

New weights will be saved in src/

Key Features
Physics-Aware Encoding

DLR Encoder extracts paracontrolled features capable of handling spatial singularities in Φ⁴₂.

Explicit Spectral Solver

A stable pseudo-spectral integration scheme separating Drift F and Diffusion G to respect the Itô SPDE structure.

Statistical Consistency Loss

Composite objective:
• L2 trajectory loss
• Log-spectral loss (Fourier-domain consistency)
• Differentiable ACF loss (temporal autocorrelation preservation)

This enforces correct roughness, long-range memory, and stationary behavior.

Contact

leduyanh1407@gmail.com
