from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import os
import torch
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnAddedKVProcessor
import numpy as np
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnAddedKVProcessor
from datasets import Dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="train_cnn")
    parser.add_argument("--decoder_dir",type=str,default=None)
    parser.add_argument("--prior_dir",type=str,default=None)
    parser.add_argument("--save_dir",type=str,default=None)
    parser.add_argument("--gpu",type=int,default=None)
    parser.add_argument("--dataset_dir",type=str,default=None)
    args = parser.parse_args()

    return args



def main():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained('kandinsky-community/kandinsky-2-2-prior', subfolder='image_encoder').to(torch.float16).to('cuda')
    unet = UNet2DConditionModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', subfolder='unet').to(torch.float16).to(f'cuda:{args.gpu}')
    prior = KandinskyV22PriorPipeline.from_pretrained('kandinsky-community/kandinsky-2-2-prior', image_encoder=image_encoder, torch_dtype=torch.float16)
    prior = prior.to(f"cuda:{args.gpu}")
    decoder = KandinskyV22Pipeline.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', unet=unet, torch_dtype=torch.float16)
    decoder = decoder.to(f"cuda:{args.gpu}")
    lora_attn_procs = {}
    d = torch.load(args.decoder_dir)
    for name in decoder.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else decoder.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = decoder.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(decoder.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = decoder.unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnAddedKVProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=4,
        ).to(f'cuda:{args.gpu}')

    decoder.unet.set_attn_processor(lora_attn_procs)
    decoder.unet.load_state_dict(d, strict=False)
    lora_attn_procs = {}
    for name in prior.prior.attn_processors.keys():
        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=2048).to(f'cuda:{args.gpu}')
    prior.prior.set_attn_processor(lora_attn_procs)
    prior.prior.load_state_dict(torch.load(args.prior_dir), strict=False)
    dataset = Dataset.from_dict(torch.load(args.dataset_dir))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(len(dataset["text"])):
        for j in range(3):
            img_emb = prior(prompt=dataset["text"][i][:77], num_inference_steps=30, num_images_per_prompt=1,)
            negative_prior_prompt ='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'
            negative_emb = prior(prompt=negative_prior_prompt, num_inference_steps=25, num_images_per_prompt=1)
            images = decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb.image_embeds,num_inference_steps=30, height=512, width=512,guidance_scale=7.5)
            image=images.images[0]
            filename = f"image_{i+1:02}_{j+1:02}.jpg"
            save_path = os.path.join(args.save_dir, filename)
            image.save(save_path)
    return






if __name__ == "__main__":
    args = parse_args()
    main()