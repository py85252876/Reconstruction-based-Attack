import argparse
import os
import torch
import torch.utils.checkpoint
from datasets import Dataset
from diffusers import StableDiffusionPipeline
from PIL import Image

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images) : return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    ) 
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=3
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--inference", type=int, default=100)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=None, torch_dtype=torch.float16,safety_checker = None,
    requires_safety_checker = False
    )
    pipeline.unet.load_attn_procs(args.output_dir)
    pipeline.to("cuda")


    dataset = Dataset.from_dict(torch.load(args.data_dir))
    for i in range(len(dataset["text"])):
        for j in range(args.num_validation_images):
            image = pipeline(dataset["text"][i], num_inference_steps=args.inference,guidance_scale=7.5).images[0]
            filename = f"image_{i+1:02}_{j+1:02}.jpg"
            save_path = os.path.join(args.save_dir, filename)
            image.save(save_path)

if __name__ == "__main__":
    main()
