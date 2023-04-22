import argparse
import json
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader

from week5.src.datasets import create_caption_db, TripletIm2Text, TripletText2Im, create_image_db
from week5.src.metrics import evaluate_im2text, evaluate_text2im
from week5.src.models import TripletNetIm2Text, EmbeddingNetImage, EmbeddingNetTextBERT, EmbeddingNetText, \
    TripletNetText2Im
from week5.src.utils_io import load_yaml_config
from week5.task_a import get_transforms


def evaluate(cfg):
    device = cfg['device']
    if cfg['type'] == 'im2text':
        # MODEL
        if cfg['text_encoder'] == 'bert':
            model = TripletNetIm2Text(EmbeddingNetImage(out_features=768),
                                      EmbeddingNetTextBERT(model_name='bert-base-uncased', out_features=768)).to(device)
        else:
            model = TripletNetIm2Text(EmbeddingNetImage(out_features=300),
                                      EmbeddingNetText(weights=cfg['model_text'], out_features=300)).to(device)
        model.load_state_dict(torch.load(cfg["model_path"]))

        transform = get_transforms()
        val_dataset = TripletIm2Text(cfg['val_captions'], cfg['val_dir'], transform=transform["val"], evaluation=True)
        captions_dict = {ann['id']: ann['caption'] for ann in val_dataset.annotations_an}
        val_dl = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
        # Compute cosine similarity between image embedding and all caption embeddings
        caption_db = create_caption_db(
            model=model,
            captions=captions_dict,
            out_path=cfg['database_path'],
            device=cfg['device']
        )
        # Evaluate
        map5, pat1, pat5 = evaluate_im2text(model, val_dl, caption_db, captions_dict, cfg['visualize'], device=device)
    elif cfg['type'] == 'text2image':
        # MODEL
        if cfg['text_encoder'] == 'bert':
            model = TripletNetText2Im(EmbeddingNetImage(out_features=768),
                                      EmbeddingNetTextBERT(model_name='bert-base-uncased', out_features=768)).to(device)
        else:
            model = TripletNetText2Im(EmbeddingNetImage(out_features=300),
                                      EmbeddingNetText(weights=cfg['model_text'], out_features=300)).to(device)
        model.load_state_dict(torch.load(cfg["model_path"]))

        transform = get_transforms()
        val_dataset = TripletText2Im(cfg['val_captions'], cfg['val_dir'], phase="val", transform=transform["val"], evaluation=True)
        im_dict = {ann['id']: ann['file_name'] for ann in val_dataset.images}
        val_dl = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
        # Compute cosine similarity between image embedding and all caption embeddings
        caption_db = create_image_db(
            model=model,
            im_dict=im_dict,
            im_dir=cfg['val_dir'],
            out_path=cfg['database_path'],
            device=cfg['device'],
            transform=transform["val"]
        )
        # Evaluate
        map5, pat1, pat5 = evaluate_text2im(model, val_dl, caption_db, im_dict, cfg['visualize'], device=device, im_dir=cfg['val_dir'])
    else:
        RuntimeError('Invalid type of evaluation')

    print(f"MAP@10: {map5:.4f}")
    print(f"PAT@1: {pat1:.4f}")
    print(f"PAT@5: {pat5:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/eval.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    evaluate(config)
