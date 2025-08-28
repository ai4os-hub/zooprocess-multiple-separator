import os, sys
import json
import numpy as np
import torch
from mtr.pytorch_detection.custom_datasets import ImageSegmentationDataset
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import shutil

from tensorboardX import SummaryWriter

def load_json(f):
    files = open(f)
    data = json.load(files)
    files.close()
    return data


def create_model_and_processor(model_type, model_pretrained=None, list_hashtags=[]):
    from transformers import MaskFormerImageProcessor

    processor = MaskFormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    import albumentations as A

    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    image_transform = A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    from transformers import Mask2FormerForUniversalSegmentation

    id2label = {int(k): v for k, v in enumerate(list_hashtags)}
    label2id = {v: k for k, v in id2label.items()}
    # model_pretrained = "/mnt/sda1/training_data/test_plancton_pano1"
    if model_pretrained == None:
        model_pretrained = "facebook/mask2former-swin-small-cityscapes-panoptic"
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_pretrained, id2label=id2label,
                                                                ignore_mismatched_sizes=True, label2id=label2id)

    return model, processor, image_transform


# define custom collate function which defines how to batch examples together
def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels,
            "mask_labels": mask_labels}


def train_model(name, model, train_dataset, image_processor, writer, batch_size=4,
                epochs=6, outdir=""):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    import torch
    from tqdm.auto import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    list_loss = list()

    running_loss = 0.0
    num_samples = 0
    for epoch in range(epochs):
        print("Epoch:", epoch)
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            loss = outputs.loss

            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            list_loss.append(running_loss / num_samples)
            writer.add_scalar("scalars/loss", (running_loss / num_samples), epoch)
            if idx % 50 == 0:
                print("Loss:", running_loss / num_samples, flush=True)

            # Optimization
            optimizer.step()

        if epoch % 2 == 0:
            path_save_model= os.path.join(outdir,name)
            model.save_pretrained(path_save_model)
            image_processor.save_pretrained(path_save_model)

    print("fin de l'entrainement")

    return list_loss


def run_train(data_dir, out_dir, device, n_epochs=10, list_hashtags=["plancton"], batch_size=16):
    from datasets import Dataset as Ds
    from datasets import Image

    writer = SummaryWriter(logdir=out_dir, flush_secs=1)

    ## Load data ----
    print('Create datasets')
    sys.stdout.flush()

    print(f"we load from local, we use data from {data_dir}")
    list_image = os.listdir(data_dir + "/images")
    list_images = [os.path.join(data_dir, "images", f) for f in list_image]
    list_label_images = [f.replace('/images', "/labels").replace("jpg", "png") for f in list_images]
    list_seg_info = [f.replace('/images', "/seg_info").replace("jpg", "json") for f in list_images]

    sys.stdout.flush()



    print(list_seg_info[:15])

    list_hashtags = ["background"] + list_hashtags

    dataset = Ds.from_dict({"image": list_images, "label": list_label_images,
                            "segments_info": [load_json(f) for f in list_seg_info]}).cast_column("image",
                                                                                                 Image()).cast_column(
        "label", Image())

    model, processor, image_transform = create_model_and_processor("mask2transform", model_pretrained=None,
                                                                   list_hashtags=list_hashtags)
    #from custom_datasets import ImageSegmentationDataset
    train_dataset = ImageSegmentationDataset(dataset, processor, transform=image_transform)

    model_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    list_loss=train_model(model_name, model, train_dataset, processor,writer=writer, batch_size=batch_size,
                epochs=n_epochs, outdir=out_dir)
    
    model_folder_path = os.path.join(out_dir, model_name)
    shutil.make_archive(model_folder_path, 'zip', model_folder_path)

    print("model trained")

    writer.close()


    return list_loss