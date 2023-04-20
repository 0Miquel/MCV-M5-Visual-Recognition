import argparse
import sys

import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader

from src.datasets import TripletIm2Text
from src.models import EmbeddingNetImage, TripletNetIm2Text, EmbeddingNetTextBERT
from src.utils_io import load_yaml_config
from week5.task_a import get_transforms
from week5.task_b import fit


def main(cfg):
    wandb.init(entity="bipulantes", project="Week5-TaskA")
    # DATASET
    transform = get_transforms()

    train_dataset = TripletIm2Text(cfg['train_captions'], cfg['train_dir'], transform=transform["train"])
    train_dl = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    val_dataset = TripletIm2Text(cfg['val_captions'], cfg['val_dir'], transform=transform["val"])
    val_dl = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    # MODEL

    # Use BERT for text embedding

    image_embedder = EmbeddingNetImage(cfg["embedding_size"]).to(cfg["device"])
    text_embedder = EmbeddingNetTextBERT(model_name='bert-base-uncased', out_features=768)
    model = TripletNetIm2Text(image_embedder, text_embedder).to(cfg["device"])

    # OPTIMIZER
    optimizer = optim.Adam(model.parameters(), cfg["lr"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
                                              steps_per_epoch=len(train_dataset) // cfg["batch_size"],
                                              epochs=cfg["num_epochs"])

    # LOSS
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    fit(train_dl, val_dl, model, optimizer, scheduler, criterion, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_c.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
