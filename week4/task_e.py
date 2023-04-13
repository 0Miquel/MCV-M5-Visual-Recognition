import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from week2.utils import load_yaml_config
from week4.losses import ContrastiveLoss
from week4.task_FASTER_MODEL import TrunkFasterRCNN
from week4.task_DATALOADER_TRIPLETS_COCO import SiameseNetworkDataset, TripletNetworkDataset
from tqdm import tqdm


def train(model, dataloader, optimizer, criterion, cfg):
    model.train()
    # Iterate throught the epochs
    for epoch in range(cfg["num_epochs"]):
        dataset_size = 0
        running_loss = 0
        # Iterate over batches
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{cfg['num_epochs']} train")
            for img0, img1, label in tepoch:
                # Send the images and labels to CUDA
                img0, img1, label = img0.to(cfg["device"]), img1.to(cfg["device"]), label.to(cfg["device"])

                # Zero the gradients
                optimizer.zero_grad()

                # Pass in the two images into the network and obtain two outputs
                output1, output2 = model(img0, img1)

                # Pass the outputs of the networks and label into the loss function
                loss_contrastive = criterion(output1, output2, label)

                # Calculate the backpropagation
                loss_contrastive.backward()

                # Optimize
                optimizer.step()

                dataset_size += output1.size(0)
                running_loss += loss_contrastive.item() * output1.size(0)
                epoch_loss = running_loss / dataset_size
                tepoch.set_postfix({"loss": epoch_loss})


def main(cfg):
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SiameseNetworkDataset(msc_ann_path=cfg["msc_ann"],
                                    transform=transformation,
                                    cfg=cfg)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers=2,
                            batch_size=8)
    model = TrunkFasterRCNN().to(cfg["device"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ContrastiveLoss()

    train(model, dataloader, optimizer, criterion, cfg)


if __name__ == "__main__":
    config_path = 'configs/task_e.yaml'
    config = load_yaml_config(config_path)

    main(config)
