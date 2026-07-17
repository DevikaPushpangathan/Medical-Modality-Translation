# CT to MRI Image Translation using Conditional GANs



## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [Generator (U-Net)](#generator-u-net)
  - [Discriminator (PatchGAN)](#discriminator-patchgan)
  - [Dual Discriminator (Optional)](#dual-discriminator-optional)
- [Loss Functions](#loss-functions)
- [Training](#training)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Setup](#setup)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project implements a Conditional Generative Adversarial Network for the task of image-to-image translation, specifically converting Computed Tomography (CT) scans to Magnetic Resonance Imaging (MRI) scans. The goal is to explore the effectiveness of GANs in synthesizing realistic medical images, which has potential applications in medical imaging analysis, data augmentation, and cross-modality image synthesis.

## Project Overview

We utilize a Pix2Pix-like architecture, comprising a U-Net based generator and a PatchGAN discriminator. The model learns a mapping from source domain images (CT) to target domain images (MRI) by jointly optimizing a adversarial loss and a L1 reconstruction loss. An dual-discriminator setup is explored to enhance the quality and stability of the generated images.

## Dataset

The model was trained on a dataset consisting of paired CT and MRI images. 

- **Source:** The dataset used is `archive (1).zip` which contains corresponding CT and MRI images. 
- **Preprocessing:** Images were resized to 128x128 pixels, converted to grayscale, and normalized to the range [-1, 1].
- **Structure:** The dataset is organized into `trainA` (CT images) and `trainB` (MRI images).

## Model Architecture

The cGAN architecture consists of a Generator and one or two Discriminators.

### Generator (U-Net)

The generator is a U-Net like architecture with skip connections. This allows the generator to effectively capture both low-level and high-level features, crucial for detailed image synthesis.

- **Encoder:** Downsampling path with convolutional layers, Batch Normalization, and LeakyReLU activations.
- **Bottleneck:** A central layer connecting the encoder and decoder.
- **Decoder:** Upsampling path with transposed convolutional layers, Batch Normalization, ReLU activations, and skip connections from the encoder.
- **Output:** A final convolutional layer with Tanh activation to output images in the normalized range.

### Discriminator (PatchGAN)

The discriminator is a PatchGAN, which classifies 70x70 pixel patches of the image as real or fake. This encourages the generator to produce high-frequency details.

- **Architecture:** Consists of several convolutional layers, Batch Normalization, and LeakyReLU activations.
- **Output:** A single-channel output representing the probability of a patch being real.

### Dual Discriminator 

In an advanced setup, two discriminators are used:

1.  **Patch Discriminator (D1):** Focuses on local image consistency.
2.  **Deep Discriminator (D2):** Uses a deeper architecture to assess global image realism. 

Both discriminators are trained with Spectral Normalization to stabilize training and improve performance.

## Loss Functions

The overall objective function for the cGAN combines an adversarial loss and an L1 reconstruction loss:

- **Adversarial Loss (GAN Loss):** Binary Cross-Entropy Loss (`nn.BCEWithLogitsLoss` for dual discriminator, `nn.BCELoss` for single discriminator) applied to the discriminator outputs to encourage the generator to produce images that are indistinguishable from real ones.
  - For the Generator: `log(D(G(x)))`
  - For the Discriminator: `log(D(y)) + log(1 - D(G(x)))`

- **L1 Reconstruction Loss:** Mean Absolute Error (`nn.L1Loss`) between the generated image and the ground truth image. This term encourages pixel-wise accuracy and prevents mode collapse. The L1 loss is weighted by a hyperparameter `L1_LAMBDA` (set to 100).
  - `L_1(G) = ||y - G(x)||_1`

- **Total Generator Loss:** `L_GAN(G) + L_1(G)`
- **Total Discriminator Loss:** `(L_D1 + L_D2) / 2` (for dual discriminator) or `(L_D_real + L_D_fake) / 2` (for single discriminator)

## Training

- **Optimizers:** Adam optimizer with `lr=2e-4`, `beta1=0.5`, `beta2=0.999` for both generator and discriminators.
- **Epochs:** Trained for 50 epochs.
- **Batch Size:** 1 (for dual discriminator setup) or 4 (for single discriminator setup).
- **Device:** Training can be performed on CPU or GPU (CUDA if available).
- **Logging:** Training losses (Generator, Discriminator, L1), SSIM, and PSNR are logged and saved to a CSV file (`metrics_log_cpu.csv`).
- **Samples:** Generated samples are saved periodically to visualize training progress.
- **Checkpoints:** Model weights are saved at each epoch, and the best model based on L1 loss is also saved.

## Results

The model demonstrates the ability to translate CT images to realistic MRI images. Visual examples and quantitative metrics show improvement over training epochs.

- **Visual Examples:** Sample images showing input CT, generated MRI, and ground truth MRI. (Include images from your `samples` directory here)
  ![Sample Output Epoch X](path/to/samples/epoch_00XX.png)

- **Loss Curves:** Plots of Generator and Discriminator losses over epochs.
  ![Loss Curves](path/to/training_output_pix2pix/loss_curves_cpu.png)

- **SSIM and PSNR Trends:** Plots showing the trend of Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) over epochs.
  ![SSIM PSNR Trends](path/to/training_output_pix2pix/ssim_psnr_cpu.png)

*Note: The images will be saved in the `samples` folder within your Google Drive output directory.* 

## Evaluation Metrics

- **SSIM (Structural Similarity Index):** Measures the structural similarity between the generated and real images. Higher values indicate better perceptual quality.
- **PSNR (Peak Signal-to-Noise Ratio):** Quantifies the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR values indicate less distortion.
- **L1 Loss:** Provides a pixel-wise measure of reconstruction accuracy.

## Usage

### Training the Model

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
    cd <YOUR_REPOSITORY_NAME>
    ```
2.  **Download Dataset:** Download the `archive (1).zip` dataset and place it in the project root directory or update the `zip_path` in the notebook.
3.  **Run the Notebook:** Execute the provided Jupyter notebook (or Python scripts) to train the model. Ensure you have the necessary dependencies installed.

### Generating Images (Inference)

Load a trained generator model and provide new CT images as input to generate MRI images.

```python
# Example of loading a generator and generating images
import torch
from torchvision.transforms import transforms
from PIL import Image

# Assuming G is your trained generator model
# Assuming transform is the same as used during training
# Assuming device is set to 'cpu' or 'cuda'

G.eval() # Set generator to evaluation mode

# Load an example CT image (replace with your path)
ct_image_path = "path/to/your/new_ct_image.png"
ct_img = Image.open(ct_image_path).convert("L")
ct_tensor = transform(ct_img).unsqueeze(0).to(device) # Add batch dimension

with torch.no_grad():
    generated_mri = G(ct_tensor)

# Post-process and save/display the generated MRI
# (e.g., denormalize, convert to PIL Image, save)
