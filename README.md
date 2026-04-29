This project implements a Denoising Diffusion Probabilistic Model (DDPM) from scratch using PyTorch to generate and reconstruct high-resolution images. It features a custom U-Net architecture with time-step embeddings and was optimized to run within a specific time budget on the CelebA-HQ dataset via Kaggle.

## Project Overview
The assignment focuses on the two core mechanics of diffusion models:

Forward Diffusion: Gradually adding Gaussian noise to an image until it becomes pure white noise.

Reverse Diffusion (Generative): Training a neural network to predict and remove that noise, effectively "creating" an image from nothing.

### Key Features
Custom U-Net Architecture: Built with residual blocks, group normalization, and sinusoidal time embeddings to handle denoising at various timesteps.

Time-Budgeted Training: Includes an adaptive preprocessing script that calculates the number of samples and epochs needed to fit within a strict 2-hour GPU window.

Mixed Precision Training: Uses torch.amp (Autocast) for faster training and reduced VRAM usage.

Image Reconstruction Task: Beyond just generating new faces, the model is tested on its ability to recover an original image after it has been partially corrupted by noise.

Quantitative Metrics: Evaluates performance using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) to measure the quality of reconstructed images.

### Tech Stack
Framework: PyTorch

Visualization: Matplotlib

Metrics: Scikit-image (PSNR, SSIM)

Deployment: Gradio (includes an app.py script for a web-based UI to interact with the model).

### Results & Visualization
The notebook automatically saves the following assets:

forward_diffusion.png: Visualizing the noise injection process.

generated_images.png: Samples created by the model from random noise.

reconstruction.png: A side-by-side comparison of Original vs. Noised vs. Recovered images.

loss_plot.png: A graph showing the MSE loss convergence over 30 epochs
