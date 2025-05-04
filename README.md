# Diffusion Networks

This repository contains two diffusion-based generative modeling projects:

1. **Simple Diffusion in 1D**  
2. **Denoising Diffusion Probabilistic Model (DDPM) for Image Generation (MNIST)**

---

## Contents

- `simple_diffusion.ipynb`  
  A minimal 1D diffusion model that learns to transform Gaussian noise into samples from a bimodal mixture of Gaussians. This notebook builds the forward and reverse processes from scratch and trains a small MLP to predict the standard normal noise realization used in the forward process.

- `DDPM_image_generation.ipynb`  
  A full DDPM implementation trained on the MNIST dataset. It uses a U-Net architecture and the `denoising-diffusion-pytorch` library to generate handwritten digits by learning to reverse the forward noising process.

---

## What Are Diffusion Models?

Diffusion models are generative models that learn to reverse a forward process that gradually corrupts data with Gaussian noise. During training, a neural network is optimized to predict the original standard normal noise `epsilon ~ N(0, I)` that was used to generate a noised input `x_t` from a clean sample `x_0`. At sampling time, the model starts from pure noise and denoises it step by step, ultimately producing a new sample from the learned distribution.

---

## Notebooks Overview

### `simple_diffusion.ipynb`

- Generates synthetic 1D data from a mixture of Gaussians
- Implements forward diffusion and reverse sampling explicitly
- Trains a neural network to predict the total standard normal noise used in generating `x_t`
- Visualizes the forward process and reverse denoising trajectory

### `DDPM_image_generation.ipynb`

- Loads and preprocesses MNIST images
- Defines a U-Net to predict the original noise realization from a noised image `x_t` and timestep `t`
- Uses the `GaussianDiffusion` class to manage the noise schedule and sampling
- Trains with mean squared error between true and predicted noise
- Generates digits from Gaussian noise using the learned reverse process

---

## Key Concepts

- **Forward Process (Markovian):**  
  Adds noise gradually over time using the formula:  
  `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`

- **Noise Prediction:**  
  The model is trained to predict `epsilon`, the total noise realization used to produce `x_t` from `x_0`. It does not predict the noise added at a single step.

- **Reverse Process:**  
  The predicted `epsilon_theta(x_t, t)` is used to compute the mean of the reverse distribution:  
  `mu_theta(x_t, t) = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t))`  
  The model denoises by sampling from a Gaussian centered at this mean.

---

## Requirements

- Python 3.8+
- torch
- torchvision
- matplotlib
- seaborn
- tqdm
- denoising-diffusion-pytorch

Install dependencies:

```bash
pip install torch torchvision matplotlib seaborn tqdm denoising-diffusion-pytorch
```

---

## Sample Outputs

- The 1D diffusion model learns to generate samples that match a bimodal Gaussian distribution.
- The DDPM generates MNIST-style digits from pure noise by reversing the learned diffusion process.

---

## References

- Ho et al., "Denoising Diffusion Probabilistic Models" (2020): https://arxiv.org/abs/2006.11239  
- lucidrains / denoising-diffusion-pytorch: https://github.com/lucidrains/denoising-diffusion-pytorch
