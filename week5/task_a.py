import argparse
import sys
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

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


def fit(dataloader, model, optimizer, scheduler, criterion, cfg):
    if cfg["mode"] == "train":
        for i in range(cfg["num_epochs"]):
            train_epoch(dataloader, model, optimizer, scheduler, criterion, cfg, i)
    elif cfg["mode"] == "evaluate":
        val_epoch(dataloader, model, criterion, cfg)


def train_epoch(dataloader, model, optimizer, scheduler, criterion, cfg, epoch):
    running_loss = 0
    dataset_size = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}/{cfg['num_epochs']} train")
        for im, cap1, cap2 in tepoch:
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


def val_epoch(dataloader, model, criterion, cfg):
    running_loss = 0
    dataset_size = 0

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Evaluating validation set")
            for im, cap1, cap2 in tepoch:
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


def main(cfg):
    # DATASET
    transform = get_transforms()
    if cfg["mode"] == "train":
        dataset = TripletIm2Text(cfg['train_captions'], cfg['train_dir'], transform=transform["train"])
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    elif cfg["mode"] == "evaluate":
        dataset = TripletIm2Text(cfg['val_captions'], cfg['val_dir'], transform=transform["val"])
        dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    # MODEL
    image_embedder = EmbeddingNetImage(cfg["embedding_size"]).to(cfg["device"])
    text_embedder = EmbeddingNetText(cfg["model_text"], cfg["embedding_size"]).to(cfg["device"])
    model = TripletNetIm2Text(image_embedder, text_embedder).to(cfg["device"])

    # OPTIMIZER
    optimizer = optim.Adam(model.parameters(), cfg["lr"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
                                              steps_per_epoch=len(dataset) // cfg["batch_size"],
                                              epochs=cfg["num_epochs"])

    # LOSS
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    fit(dataloader, model, optimizer, scheduler, criterion, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_a.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
