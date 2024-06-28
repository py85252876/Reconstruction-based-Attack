from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as ds
from transformers import AutoProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch
import argparse
from huggingface_hub import notebook_login
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune blip")
    parser.add_argument("--data_dir",nargs='+',type=str,default=None)
    parser.add_argument("--tokenid",type=str,default=None)
    args = parser.parse_args()

    return args

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset["text"])

    def __getitem__(self, idx):
        encoding = self.processor(images=self.dataset["image"][idx], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = self.dataset["text"][idx]
        return encoding



def main():
    notebook_login()
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    def collate_fn(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch
    dataset = {"text":[],"image":[]}
    for item in args.data_dir:
        temp = ds.from_dict(torch.load(item))
        dataset["text"].extend(temp["text"])
        dataset["image"].extend(temp["image"])
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=96, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    access_token=args.tokenid
    model.train()
    model.to(device)
    progress_bar = tqdm(range(100, 500))
    progress_bar.set_description("Epoch")
    for epoch in range(500):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            loss = outputs.loss

            print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        if epoch % 100 == 0:
            model.push_to_hub(f"blip-wit-1000-{epoch}",token=access_token)
        progress_bar.update(1)
    image = dataset["image"][0]
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=25)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    access_token=args.tokenid
    model.push_to_hub("blip-wit-1000-500",token=access_token)


if __name__ == "__main__":
    args = parse_args()
    main()
