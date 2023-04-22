import argparse
import json
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader

from week5.src.datasets import create_caption_db, TripletIm2Text
from week5.src.metrics import evaluate_im2text
from week5.src.models import TripletNetIm2Text, EmbeddingNetImage, EmbeddingNetTextBERT, EmbeddingNetText
from week5.src.utils_io import load_yaml_config
from week5.task_a import get_transforms


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg['type'] == 'text2image':
        # MODEL
        if cfg['text_encoder'] == 'bert':
            model = TripletNetIm2Text(EmbeddingNetImage(out_features=768),
                                      EmbeddingNetTextBERT(model_name='bert-base-uncased', out_features=768)).to(device)
        else:
            model = TripletNetIm2Text(EmbeddingNetImage(out_features=300),
                                      EmbeddingNetText(weights=cfg['model_text'], out_features=300)).to(device)
        model.load_state_dict(torch.load(cfg["model_path"]))

        # Load captions for the dataset
        with open(cfg['val_captions'], 'r') as f:
            annotations = json.load(f)
            captions = {ann['id']: ann['caption'] for ann in annotations['annotations']}

        transform = get_transforms()
        val_dataset = TripletIm2Text(cfg['val_captions'], cfg['val_dir'], transform=transform["val"], evaluation=True)
        val_dl = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
        # Compute cosine similarity between image embedding and all caption embeddings
        caption_db = create_caption_db(model=model, captions=captions, out_path=cfg['database_path'], device=device)
        # Evaluate
        evaluate_im2text(model, val_dl, caption_db, captions, cfg['visualize'])
    elif cfg['type'] == 'image2text':
        pass
    else:
        RuntimeError('Invalid type of evaluation')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/export/home/group02/MCV-M5-Visual-Recognition/week5/configs/eval.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    evaluate(config)
