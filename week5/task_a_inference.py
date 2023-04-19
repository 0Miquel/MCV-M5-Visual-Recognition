import json

import torch
import numpy as np
import argparse
import sys
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from src.models import EmbeddingNetImage, EmbeddingNetText, TripletNetIm2Text
from src.utils_io import load_yaml_config
from src.datasets import TripletIm2Text, TripletText2Im


def get_transforms():
    augmentations = {
        "train":
            transforms.Compose([
                transforms.ColorJitter(brightness=.3, hue=.3),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        "val":
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    return augmentations


def image2text_inference(image_path, model, captions):
    transform = get_transforms()["val"]
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)#.to("cuda")
    image_embedding = model.get_embedding_image(image_tensor)

    # Compute cosine similarity between image embedding and all caption embeddings
    caption_embeddings = []
    for caption in captions:
        caption_embedding = model.get_embedding_text([caption])[0]
        caption_embeddings.append(caption_embedding)
    caption_embeddings = torch.Tensor(caption_embeddings)#.to(model.device)
    sim_scores = torch.nn.functional.cosine_similarity(image_embedding, caption_embeddings)

    # Get index of the top 5 captions with less distances in sim scores
    top5 = np.argsort(sim_scores.cpu().detach().numpy())[-5:]
    top5_captions = [captions[i] for i in top5]

    last5 = np.argsort(sim_scores.cpu().detach().numpy())[:5]
    last5_captions = [captions[i] for i in last5]
    return top5_captions, last5_captions


def main(cfg):
    # MODEL
    model = TripletNetIm2Text(EmbeddingNetImage(out_features=300),
                              EmbeddingNetText(weights=cfg["fasttext_path"], out_features=300))
    model.load_state_dict(torch.load(cfg["save_path"]))

    # Load captions for the dataset
    with open(cfg['val_captions'], 'r') as f:
        annotations = json.load(f)
        captions = [ann['caption'] for ann in annotations['annotations']]

    # Predict caption for a new image
    image_path = cfg["val_dir"]+'COCO_val2014_000000000073.jpg'
    predicted_caption, lastcaptions = image2text_inference(image_path, model, captions)
    print("predicted: ",predicted_caption)
    print("worst captions: ",lastcaptions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_a.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
