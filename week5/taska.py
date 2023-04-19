import os

from torch.optim.lr_scheduler import ExponentialLR

from week5.i_o import load_yaml_config
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import umap
from cycler import cycler
import torch
import logging
import argparse
import sys
from torch import optim
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch import nn
import torchvision.datasets as datasets
from models import ImgEncoder, TextEncoder
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils import decay_learning_rate, mpk
from datasets import CocoCaptionsWithNegative
import torchtext


def main(cfg):
    loss_func = nn.TripletMarginLoss(p=2)

    # Define transforms to preprocess the images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load COCO dataset
    # train_dataset = datasets.CocoCaptions(root=cfg["train_path"],
    #                                       annFile=cfg["train_captions_path"],
    #                                       transform=transform)
    # val_dataset = datasets.CocoCaptions(root=cfg["val_path"],
    #                                       annFile=cfg["val_captions_path"],
    #                                       transform=transform_val)

    train_dataset = CocoCaptionsWithNegative(root=cfg["train_path"], annFile=cfg["train_captions_path"],
                                             transform=transform)
    val_dataset = CocoCaptionsWithNegative(root=cfg["train_path"], annFile=cfg["train_captions_path"],
                                             transform=transform_val)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=True)



    # TEXT & IMGS MODELS
    image_model = ImgEncoder()
    text_model = TextEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.to(device)
    text_model.to(device)
    # init weights
    image_model.init_weights()
    text_model.init_weights()

    if cfg["train"]:
        # optimizer
        params = list(image_model.parameters())
        params += list(text_model.parameters())

        optimizer = Adam(params, lr=cfg["lr"])

        # training loop
        for epoch in range(cfg["num_epochs"]):
            decay_learning_rate(cfg["lr"], optimizer, epoch)

            for i, img_triple in enumerate(train_dataloader):

                # execute image_triple
                img_features, pos_text_features_tuple, neg_text_features_tuple = img_triple


                img_features = img_features.to(device)

                # define a field to handle the text data
                text_field = torchtext.data.Field(tokenize='spacy')


                pos_text_features = []
                for pos_caption in pos_text_features_tuple:
                    pos_text_features.append(text_field.process([text_field.tokenize(pos_caption)]).to(device))
                neg_text_features = []
                for neg_caption in neg_text_features_tuple:
                    neg_text_features.append(text_field.process([text_field.tokenize(neg_caption)]).to(device))

                pos_text_features = torch.stack(pos_text_features, dim=0)
                neg_text_features = torch.stack(neg_text_features, dim=0)

                image_encoded = image_model(img_features)
                pos_text_encoded = text_model(pos_text_features)
                neg_text_encoded = text_model(neg_text_features)

                loss = loss_func(image_encoded, pos_text_encoded, neg_text_encoded)

                optimizer.zero_grad()
                loss.backward()
                if cfg["grad_clip"] > 0: #avoid exploting gradients
                    clip_grad_norm_(params, cfg["grad_clip"])
                optimizer.step()

                print(f'epoch: {epoch}\titeration: {i}\tLoss: {loss}')

        state_dict = [image_model.state_dict(), text_model.state_dict()]
        os.makedirs("weights", exist_ok=True)
        torch.save(state_dict, cfg["save_weights"])

    else:
        # inference with trained weights
        state_dict = torch.load(cfg["save_weights"])
        image_model.load_state_dict(state_dict[0])
        text_model.load_state_dict(state_dict[1])

        # optimizer
        params = list(image_model.parameters())
        params += list(text_model.parameters())

        optimizer = Adam(params, lr=cfg["lr"])
        scheduler = ExponentialLR(optimizer, args.gamma)

        image_model.train()
        text_model.train()

        p1, p5 = validate(train_dataloader, image_model, text_model, args.anchor, -1, 'train')
        p1, p5 = validate(val_dataloader, image_model, text_model, args.anchor, -1, 'validation')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/taska.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
