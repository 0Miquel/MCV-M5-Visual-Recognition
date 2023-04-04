import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from week4.i_o import load_yaml_config
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
from models import HeadlessResnet, Embedder, Fusion
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from week4.utils import mAP, mpk, mrk, plot_roc, plot_imgs, plot_pr


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


def infere_dataset(dataset, embedder_size, model):
    data = np.empty((len(dataset), embedder_size))
    with torch.no_grad():
        model.eval()
        for ii, (img, _) in tqdm(enumerate(dataset)):
            data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()
    return data


def main(cfg):
    # set datasets
    augmentations = get_transforms()
    train_dataset = ImageFolder(cfg["train_path"], transform=augmentations["train"])
    val_dataset = ImageFolder(cfg["val_path"], transform=augmentations["val"])

    # set inference model
    trunk = HeadlessResnet(cfg["trunk_path"], True)
    embedder = Embedder(512, cfg["embedder_size"], cfg["embedder_path"])
    inference_model = Fusion(trunk, embedder)

    # set data
    catalogue_meta = [(x[0].split('/')[-1], x[1]) for x in train_dataset.imgs]
    query_meta = [(x[0].split('/')[-1], x[1]) for x in val_dataset.imgs]

    catalogue_data = infere_dataset(train_dataset, cfg['embedder_size'], inference_model)
    query_data = infere_dataset(val_dataset, cfg['embedder_size'], inference_model)

    catalogue_labels = np.asarray([x[1] for x in catalogue_meta])
    query_labels = np.asarray([x[1] for x in query_meta])

    # Image retrieval:
    knn = KNeighborsClassifier(n_neighbors=cfg["k_neighbours"], p=1)
    knn = knn.fit(catalogue_data, catalogue_labels)

    plot_roc(catalogue_data, query_data, catalogue_labels, query_labels, knn)
    plot_pr(catalogue_data, query_data, catalogue_labels, query_labels, knn)

    neighbors = knn.kneighbors(query_data)[1]
    neighbors_labels = []
    for i in range(len(neighbors)):
        neighbors_class = [catalogue_meta[j][1] for j in neighbors[i]]
        neighbors_labels.append(neighbors_class)

    query_labels = [x[1] for x in query_meta]

    p_1 = mpk(query_labels, neighbors_labels, 1)
    p_5 = mpk(query_labels, neighbors_labels, 5)
    p_10 = mpk(query_labels, neighbors_labels, 10)
    print(f"Precision@1={p_1}, Precision@5={p_5}, Precision@10={p_10}")

    r_1 = mrk(query_labels, neighbors_labels, 1)
    r_5 = mrk(query_labels, neighbors_labels, 5)
    r_10 = mrk(query_labels, neighbors_labels, 10)
    print(f"Recall@1={r_1}, Recall@5={r_5}, Recall@10={r_10}")

    map_ = mAP(query_labels, neighbors_labels)
    print(f"mAP={map_}")

    plot_imgs(neighbors, query_meta, catalogue_meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/inference_b.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
