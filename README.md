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
    - [Fine-tune Image Captioning Models](#fine-tune-image-captioning-models)
- [Generate Images from Models](#generate-images-from-models)
    - [Generate Images with Captioning Models](#generate-images-with-captioning-models)
- [Calculate Reconstruction Distance](#calculate-reconstruction-distance)
- [Test Attack Accuracy](#test-attack-accuracy)

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

In our research, both **Attack-II** and **Attack-IV** scenarios operate without using image captions to query the model. Therefore, the captioning model requires fine-tuning on an auxiliary dataset.

```bash
python3 blip_finetune.py --data_dir auxiliary-dataset-dir
```

## Generate Images from Models

After fine-tuning the Stable Diffusion model, we proceeded to generate images using this refined model.

```bash
accelerate launch --gpu_ids 0 --main_process_port=28500 inference.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--num_validation_images=3 \
--inference=30 \
--output_dir=checkpoints-dir \
--data_dir=dataset-dir \
--save_dir=save-dir \
--seed=1337
```

### Generate Images with Captioning Models

Both **Attack-II** and **Attack-IV** require the assistance of captioning models to generate images.

```bash
python3 build_caption.py \
--data_dir=data-dir \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--output_dir=checkpoints-dir \
--seed=1337 \
--inference_step=30 \
--model_id=model-id \
--num_validation_images=3 \
--save_dir=save-dir \
--gpu_id=0
```

## Calculate Reconstruction Distance

Calculate the reconstruction distance between the generated images and the query image.

```bash
python3 cal_embedding.py \
--data_dir=data-dir \
--sample_file=image-save-dir \
--membership=0 \
--img_num=3 \
--gpu=0 \
--save_dir=distance-save-dir
```

## Test Attack Accuracy

Utilize the reconstruction distance to train an inference model for predicting the membership of the query data.

```bash
python3 test_accuracy.py \
--target_member_dir=target-member-dir \
--target_non-member_dir=target-non_member-dir \
--shadow_member_dir=shadow-member-dir \
--shadow_non-member_dir=shadow-non_member-dir \
--method="classifier"
```
