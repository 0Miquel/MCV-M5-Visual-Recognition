"""
This code has been written following the tutorial from Pytorch Metric learning
https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/MetricLossOnly.ipynb
"""

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import distances, reducers
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch

from week4.i_o import load_yaml_config

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import umap
from cycler import cycler

import logging
import argparse
import sys
from torch import optim
from models import HeadlessResnet, Embedder, Fusion
from torchvision.datasets import ImageFolder

from torchvision.datasets import CocoDetection
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn
import torch.optim as optim
from contrastive_loss import ContrastiveLoss

import wandb


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    idx_to_class = {0: 'Opencountry', 1: 'coast', 2: 'forest', 3: 'highway', 4: 'inside_city',
                    5: 'mountain', 6: 'street', 7: 'tallbuilding'}

    logging.info(
        f"UMAP plot for the {split_name} split and epoch {args[0]}"
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        ax.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=10,
                label=f"{idx_to_class[label_set[i]]}")
    plt.legend(loc='best', fontsize='large', markerscale=1)
    plt.title(f"UMAP plot for the {split_name} split and epoch {args[0]}")
    plt.savefig('umap.jpg')
    plt.show()


def get_transforms():
    augmentations = {
        "train":
            transforms.Compose([
                transforms.ColorJitter(brightness=.3, hue=.3),
                transforms.RandomResizedCrop(256, (0.15, 1.0)),
                transforms.RandomRotation(degrees=30),
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


def main(cfg):
    # wandb.tensorboard.patch(root_logdir="example_tensorboard")
    # wandb.init(project='test', sync_tensorboard=True)

    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)

    # set datasets
    augmentations = get_transforms()
    train_dataset = CocoDetection(root=cfg["train_path"], annFile=cfg["train_ann_file"], transform=augmentations)
    val_dataset = CocoDetection(root=cfg["val_path"], annFile=cfg["val_ann_file"], transform=augmentations)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,num_workers=4)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    # train_dataset = ImageFolder(cfg["train_path"], transform=augmentations["train"])
    # val_dataset = ImageFolder(cfg["val_path"], transform=augmentations["val"])


    # set model
    backbone = models.resnet50(pretrained=True)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
    model = FasterRCNN(backbone, num_classes=91)
    criterion = ContrastiveLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    # define the data loader and dataset
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    # training loop
    for epoch in range(cfg["num_epochs"]):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_e.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
