# Reconstruction-based Attack

This work presents an effective black-box membership inference attack framework tailored for the latest generation of image generator models. This novel framework exploits the memorization characteristics of image generator models regarding training data to mount attacks, utilizing reconstruction distance as indicative features.

In this repository, we can find:

1. Using LoRA to fine-tune Stable Diffusion on a customized dataset.
2. Generating images from the fine-tuned models based on different attack scenarios.
3. Calculating the reconstruction distance between the generated and query images.
4. Using the reconstruction distance as features to train an inference model, including **Threshold-based**, **Distribution-based**, and **Classifier-based** approaches.

## Table of Contents

- [Download Dependencies](#download-dependencies)
	- [DDPM dependencies](#ddpm-dependencies)
	- [Imagen dependencies](#imagen-dependencies)
- [Prepare Datasets](#prepare-datasets)

## Environment Setup

> Before running the code, make sure to install all dependency files.

Install [requirements.txt](./requirements.txt) and run:

```bash
pip install -r requirements.txt
```
