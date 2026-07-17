# Improving Medical Image Modality Translation with Dual Discriminator Pix2Pix Networks: A CT-to-MRI Study

## Overview

This project presents an enhanced **Pix2Pix Conditional GAN (cGAN)** framework for translating **Computed Tomography (CT)** images into **Magnetic Resonance Imaging (MRI)** images. The proposed architecture introduces a **dual-discriminator** design to improve both local texture generation and global anatomical consistency, resulting in more realistic synthetic MRI images.

The work was developed as part of my **Master of Science in Data Science** dissertation at **Vellore Institute of Technology (VIT), Chennai**. :contentReference[oaicite:0]{index=0}

---

## Motivation

MRI provides superior soft tissue contrast compared to CT, making it essential for diagnosing neurological disorders. However, MRI acquisition is:

- Expensive
- Time-consuming
- Less accessible
- Unsuitable for certain patients with contraindications

This project aims to generate high-quality MRI images directly from CT scans using deep learning, potentially improving clinical workflows when MRI is unavailable. :contentReference[oaicite:1]{index=1}

---

## Proposed Architecture

The proposed model consists of:

- **Generator**
  - U-Net architecture
  - Skip connections for preserving anatomical structures

- **Discriminators**
  - PatchGAN Discriminator
    - Learns local textures and fine details
  - Global (Deep) Discriminator
    - Preserves overall anatomical consistency

The generator receives a CT image as input and synthesizes its corresponding MRI image while both discriminators jointly guide the adversarial learning process. :contentReference[oaicite:2]{index=2}

---

## Features

- CT → MRI image translation
- Dual-discriminator Pix2Pix architecture
- U-Net generator with skip connections
- Structural preservation of anatomical regions
- Quantitative evaluation using SSIM and PSNR
- Training visualization for Generator and Discriminator losses

---

## Dataset

- Paired CT–MRI brain image dataset
- Images resized to **128 × 128**
- Pixel normalization to **[-1, 1]**
- Dataset split into:
  - Training
  - Validation
  - Testing

:contentReference[oaicite:3]{index=3}

---

## Evaluation Metrics

The model is evaluated using:

- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- L1 Loss
- Generator Loss
- Discriminator Loss

These metrics assess structural fidelity, reconstruction quality, and training stability. :contentReference[oaicite:4]{index=4}

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PIL
- scikit-image
- Google Colab

:contentReference[oaicite:5]{index=5}

---

## Project Structure

```
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── generator.py
│   ├── patch_discriminator.py
│   └── global_discriminator.py
│
├── training/
│   ├── train.py
│   └── losses.py
│
├── evaluation/
│   ├── metrics.py
│   └── inference.py
│
├── results/
│   ├── generated_images/
│   ├── loss_curves/
│   └── comparison_images/
│
├── notebooks/
├── README.md
└── requirements.txt
```

*(Modify the structure above according to your repository.)*

---

## Results

The proposed dual-discriminator model demonstrated:

- Improved structural similarity (SSIM)
- Higher PSNR
- Lower L1 reconstruction error
- Stable adversarial training
- Better preservation of anatomical structures than the conventional single-discriminator Pix2Pix model

Both quantitative metrics and qualitative visual comparisons indicate improved MRI synthesis quality. :contentReference[oaicite:6]{index=6}

---

## Limitations

Current limitations include:

- Limited paired CT–MRI dataset
- Training for only 42 epochs
- 2D slice-based learning without volumetric consistency
- Reduced preservation of fine anatomical textures

Future work includes:

- Larger datasets
- 3D medical image translation
- Hybrid perceptual and structural loss functions
- Multi-scale adversarial learning
- Diffusion-based medical image translation

:contentReference[oaicite:7]{index=7}

---

## Future Improvements

- 3D volumetric CT-to-MRI synthesis
- Diffusion model-based image translation
- Attention-enhanced U-Net
- Clinical validation by radiologists
- Foundation model integration for medical imaging

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{devika2025ctmri,
  title={Improving Medical Image Modality Translation with Dual Discriminator Pix2Pix Networks: A CT-to-MRI Study},
  author={Devika P},
  school={Vellore Institute of Technology, Chennai},
  year={2025}
}
```

---

## Author

**Devika P**

M.Sc. Data Science  
Vellore Institute of Technology, Chennai

**Research Interests**

- Medical Image Analysis
- Computer Vision
- Generative AI
- Diffusion Models
- Deep Learning


## License

This project is intended for academic and research purposes.
