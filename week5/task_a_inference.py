import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models import EmbeddingNetImage, EmbeddingNetText, TripletNetIm2Text, EmbeddingNetTextBERT
from src.utils_io import load_yaml_config
from week5.task_a import get_transforms


def image2text_inference(image_path, model, captions):
    transform = get_transforms()["val"]
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # .to("cuda")
    image_embedding = model.get_embedding_image(image_tensor)

    # Compute cosine similarity between image embedding and all caption embeddings
    caption_embeddings = []
    for caption in tqdm(captions):
        caption_embedding = model.get_embedding_text([caption])[0]
        caption_embeddings.append(caption_embedding)
    caption_embeddings = torch.Tensor(caption_embeddings)  # .to(model.device)
    sim_scores = torch.nn.functional.cosine_similarity(image_embedding, caption_embeddings)

    # Get index of the top 5 captions with the closest distances in sim scores
    top5 = np.argsort(sim_scores.cpu().detach().numpy())[-5:]
    top5_captions = [captions[i] for i in top5]

    last5 = np.argsort(sim_scores.cpu().detach().numpy())[:5]
    last5_captions = [captions[i] for i in last5]
    return top5_captions, last5_captions


def main(cfg):
    # MODEL
    model = TripletNetIm2Text(EmbeddingNetImage(out_features=768),
                              EmbeddingNetTextBERT(model_name='bert-base-uncased', out_features=768))
    model.load_state_dict(torch.load(cfg["save_path"]))

    # Load captions for the dataset
    with open(cfg['val_captions'], 'r') as f:
        annotations = json.load(f)
        captions = [ann['caption'] for ann in annotations['annotations']]

    # Predict caption for a new image
    image_path = cfg["val_dir"] + 'COCO_val2014_000000000073.jpg'
    predicted_caption, last_captions = image2text_inference(image_path, model, captions)
    print("predicted: ", predicted_caption)
    print("worst captions: ", last_captions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_a.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
