import argparse
import sys
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import wandb

from src.models import EmbeddingNetImage, EmbeddingNetText, TripletNetIm2Text, EmbeddingNetTextBERT
from src.utils_io import load_yaml_config
from src.datasets import TripletIm2Text


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


def fit(train_dl, val_dl, model, optimizer, scheduler, criterion, cfg):
    os.makedirs("models", exist_ok=True)
    best_loss = 100000000
    for i in range(cfg["num_epochs"]):
        train_metrics = train_epoch(train_dl, model, optimizer, scheduler, criterion, cfg, i)
        val_metrics = val_epoch(val_dl, model, criterion, cfg, i)
        # save model if new is best
        new_loss = val_metrics["val/loss"]
        if new_loss < best_loss:
            best_loss = new_loss
            torch.save(model.state_dict(), cfg["save_path"])
        # log metrics
        train_metrics.update(val_metrics)
        wandb.log(train_metrics)


def train_epoch(dataloader, model, optimizer, scheduler, criterion, cfg, epoch):
    running_loss = 0
    dataset_size = 0
    count = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}/{cfg['num_epochs']} train")
        for im, cap1, cap2 in tepoch:
            count += 1
            im = im.to(cfg["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            feat_im, feat_cap1, feat_cap2 = model(im, cap1, cap2)
            feat_cap1 = torch.tensor(feat_cap1, dtype=torch.float32).to(cfg["device"])
            feat_cap2 = torch.tensor(feat_cap2, dtype=torch.float32).to(cfg["device"])
            # loss
            loss = criterion(feat_im, feat_cap1, feat_cap2)
            # backward
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # compute epoch loss and current learning rate
            dataset_size += im.size(0)
            running_loss += loss.item() * im.size(0)
            epoch_loss = running_loss / dataset_size
            current_lr = optimizer.param_groups[0]['lr']
            tepoch.set_postfix({"loss": epoch_loss, "lr": current_lr})

            # if count == 10:
            #     break

    return {"train/loss": epoch_loss, "train/lr": current_lr}


def val_epoch(dataloader, model, criterion, cfg, epoch):
    running_loss = 0
    dataset_size = 0
    count = 0
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{cfg['num_epochs']} val")
            for im, cap1, cap2 in tepoch:
                count += 1
                im = im.to(cfg["device"])
                # forward
                feat_im, feat_cap1, feat_cap2 = model(im, cap1, cap2)
                feat_cap1 = torch.tensor(feat_cap1, dtype=torch.float32).to(cfg["device"])
                feat_cap2 = torch.tensor(feat_cap2, dtype=torch.float32).to(cfg["device"])
                # loss
                loss = criterion(feat_im, feat_cap1, feat_cap2)

                # compute epoch loss
                dataset_size += im.size(0)
                running_loss += loss.item() * im.size(0)
                epoch_loss = running_loss / dataset_size
                tepoch.set_postfix({"loss": epoch_loss})

                # if count == 10:
                #     break

    return {"val/loss": epoch_loss}


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
