"""
This code has been written following the tutorial
https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/"""
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from week4.i_o import load_yaml_config


class SiameseNetworkDataset(Dataset):
    def __init__(self, msc_ann_path, transform=None, cfg=None):
        self.msc_ann_path = msc_ann_path
        self.transform = transform
        self.cfg = cfg

    def __getitem__(self, index):
        with open(self.msc_ann_path) as f:
            msc_ann = json.load(f)

        classes = list(msc_ann['train'].keys())
        class_idx = np.random.randint(len(classes))
        selected_class = classes[class_idx]

        selected_images = np.random.choice(msc_ann['train'][selected_class], size=2, replace=False)

        image_pair = []
        for image_path in selected_images:
            path_name = self.cfg["train_path"] + "COCO_train2014_" + "{:012d}".format(image_path) + ".jpg"
            image = Image.open(path_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            image_pair.append(image)

        label = int(class_idx == np.random.randint(len(classes)))

        return (image_pair[0], image_pair[1], label)

    def __len__(self):
        with open(self.msc_ann_path) as f:
            msc_ann = json.load(f)

        num_pairs = sum(len(image_paths) for image_paths in msc_ann['train'].values()) * 2

        return num_pairs

class TripletNetworkDataset(Dataset):
    def __init__(self, msc_ann_path, transform=None, cfg=None):
        self.msc_ann_path = msc_ann_path
        self.transform = transform
        self.cfg = cfg
        self.msc_ann = json.load(open(self.msc_ann_path))
        self.classes = list(self.msc_ann['train'].keys())

    def __getitem__(self, index):


        class_idx = np.random.randint(len(self.classes))
        selected_class = self.classes[class_idx]

        # Select anchor image from selected class
        anchor_path = np.random.choice(self.msc_ann['train'][selected_class])
        anchor = self._load_image(anchor_path)

        # Select positive image from same class as anchor
        positive_path = np.random.choice(self.msc_ann['train'][selected_class])
        positive = self._load_image(positive_path)

        # Select negative image from a different class
        negative_class_idx = (class_idx + np.random.randint(1, len(self.classes))) % len(self.classes)
        negative_class = self.classes[negative_class_idx]
        negative_path = np.random.choice(self.msc_ann['train'][negative_class])
        negative = self._load_image(negative_path)

        return (anchor, positive, negative)

    def __len__(self):
        with open(self.msc_ann_path) as f:
            msc_ann = json.load(f)

        num_triplets = sum(len(image_paths) for image_paths in msc_ann['train'].values())

        return num_triplets

    def _load_image(self, path):
        path_name = self.cfg["train_path"] + "COCO_train2014_" + "{:012d}".format(path) + ".jpg"
        image = Image.open(path_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image



# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imshow_triplet(img1, img2, img3, text=None):
    npimg1 = img1.numpy()
    npimg2 = img2.numpy()
    npimg3 = img3.numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.imshow(np.transpose(npimg1, (1, 2, 0)))
    ax1.axis("off")
    ax2.imshow(np.transpose(npimg2, (1, 2, 0)))
    ax2.axis("off")
    ax3.imshow(np.transpose(npimg3, (1, 2, 0)))
    ax3.axis("off")

    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.show()


# Plotting data
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

def main(cfg):


    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                         transforms.ToTensor()
                                         ])

    # Initialize the network
    siamese_dataset = TripletNetworkDataset(msc_ann_path=cfg["msc_ann"],
                                            transform=transformation,
                                            cfg=cfg)

    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=2,
                                batch_size=8)

    example_batch = next(iter(vis_dataloader))

    #plot all the images
    for i in range(8):
        imshow_triplet(example_batch[0][i], example_batch[1][i], example_batch[2][i])
    print(example_batch[1].numpy().reshape(-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_e.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
