# Reconstruction-based Attack

This work presents an effective black-box membership inference attack framework tailored for the latest generation of image generator models. This novel framework exploits the memorization characteristics of image generator models regarding training data to mount attacks, utilizing reconstruction distance as indicative features.

In this repository, we can find:

1. Using LoRA to fine-tune Stable Diffusion on a customized dataset.
2. Generating images from the fine-tuned models based on different attack scenarios.
3. Calculating the reconstruction distance between the generated and query images.
4. Using the reconstruction distance as features to train an inference model, including **Threshold-based**, **Distribution-based**, and **Classifier-based** approaches.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Fine-tune image generator](#fine-tune-image-generator-models)

## Environment Setup

> Before running the code, make sure to install all dependency files.

Install [requirements.txt](./requirements.txt) and run:

```bash
pip install -r requirements.txt
```
And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

## Fine-tune Image Generator Models

After preparing the dataset, we employed the [🤗diffusers](https://github.com/huggingface/diffusers/) train_text_to_image_lora.py to fine-tune the Stable Diffusion v1-5.

```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir=prepare_dataset \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --project_name="SD v1-5" \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=62500 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=output_dir \
  --report_to=wandb \
  --resume_from_checkpoint="latest" \
  --checkpointing_steps=12500 \
  --validation_prompt=valid_prompt \
  --seed=1337
```

### Fine-tune Image Captioning Models

In our research, both Attack-II and Attack-IV scenarios operate without using image captions to query the model. Therefore, the captioning model requires fine-tuning on an auxiliary dataset.

```bash
python3 blip_finetune.py --data_dir auxiliary_dataset_dir
```

## Generate Images from Models

### Generate Images with Captioning Models

## Calculate Reconstruction Distance

## Test Attack Accuracy
