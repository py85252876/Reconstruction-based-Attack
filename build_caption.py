import torch
from PIL import Image
from datasets import Dataset
import argparse
from accelerate import Accelerator
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import os
from transformers import BlipForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images) : return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune blip2")
    parser.add_argument("--data_dir",type=str,default=None)
    parser.add_argument("--output_dir",type=str,default=None)
    parser.add_argument("--pretrained_model_name_or_path",type=str,default=None)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--num_validation_images",type=int,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--model_id",type=str,default=None)
    parser.add_argument("--inference_step",type=int,default=None)
    parser.add_argument("--gpu_id",type=int,default=None)
    args = parser.parse_args()

    return args

def prepare():
    
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to(device)
    accelerator = Accelerator()

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=None, torch_dtype=torch.float16,safety_checker = None,
    requires_safety_checker = False
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.load_attn_procs(args.output_dir)
    pipeline.to(device)
    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    return device, processor, model, pipeline, generator

def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dataset = Dataset.from_dict(torch.load(args.data_dir))
    for i in range(len(dataset["image"])):
        inputs = processor(images=dataset["image"][i], return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=77)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        for j in range(args.num_validation_images):
            image = pipeline(generated_caption, num_inference_steps=args.inference_step, generator=generator,guidance_scale=7.5).images[0]
            filename = f"image_{i+1:02}_{j+1:02}.jpg"
            save_path = os.path.join(args.save_dir, filename)
            image.save(save_path)
    
    


if __name__ == "__main__":
    args = parse_args()
    device, processor, model, pipeline, generator = prepare()
    main()