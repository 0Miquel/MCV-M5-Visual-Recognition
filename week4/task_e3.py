import argparse
import json
import os
import random
import sys
from collections import OrderedDict
from typing import Dict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from week4.i_o import load_yaml_config
from week4.losses import ContrastiveLoss


class SiameseCOCODataset(Dataset):
    def __init__(self, cfg: Dict, phase: str = 'train', transform=None, inference: bool = False):
        self.mcv_ann_path = cfg["mcv_ann"]
        self.phase = phase
        self.transform = transform
        self.mcv_ann = json.load(open(self.mcv_ann_path))[phase]
        self.classes = list(self.mcv_ann.keys())

        self.imgs_dir = cfg[phase + "_path"]
        self.imgs_to_load = []
        for img_ids in self.mcv_ann.values():
            for img_id in img_ids:
                if img_id not in self.imgs_to_load:
                    self.imgs_to_load.append(img_id)
        self.imgs_path = [
            self.imgs_dir + f"COCO_{os.path.basename(os.path.normpath(self.imgs_dir))}_" + "{:012d}".format(
                img_to_load) + ".jpg" for img_to_load in self.imgs_to_load]
        self.inference = inference

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        img_id = int(img_path.split("/")[-1].split(".")[0].split("_")[-1])
        img = cv2.imread(img_path)[:, :, ::-1]
        transformed_img = self.transform(image=img)["image"]

        if self.inference:
            return transformed_img, self._load_annot(img_id), img_path

        classes = [class_id for class_id, img_ids in self.mcv_ann.items() if img_id in img_ids]

        mcv_ann_negative = {class_id: img_ids for class_id, img_ids in self.mcv_ann.items() if class_id not in classes}
        mcv_ann_positive = {class_id: img_ids for class_id, img_ids in self.mcv_ann.items() if class_id in classes}

        if not classes:
            should_get_same_class = 0
            negative_union_imgs = set().union(*mcv_ann_negative.values())
            pick_img_id = random.choice(tuple(negative_union_imgs))
        else:
            should_get_same_class = random.randint(0, 1)
            # imgs that have at least one of the classes
            positive_union_imgs = set().union(*mcv_ann_positive.values())
            if should_get_same_class:
                pick_img_id = random.choice(tuple(positive_union_imgs))
            else:  # should_get_same_class == 0 <-- different class
                # imgs that don't have at least one of the classes
                negative_union_imgs = set().union(*mcv_ann_negative.values())
                negative_imgs = negative_union_imgs - positive_union_imgs
                pick_img_id = random.choice(tuple(negative_imgs))

        pick_img_path = self.imgs_dir + f"COCO_{os.path.basename(os.path.normpath(self.imgs_dir))}_" + "{:012d}".format(
            pick_img_id) + ".jpg"
        pick_img = cv2.imread(pick_img_path)[:, :, ::-1]
        transformed_pick_img = self.transform(image=pick_img)["image"]

        return transformed_img, transformed_pick_img, should_get_same_class

    def _load_annot(self, img_id: int) -> Dict[int, int]:
        """
        Returns a dictionary with the annotations of the image
        The keys are the category ids and the values are the number of instances of that category
        :param img_id: id of the image
        :return: dictionary with the annotations of the image
        """
        annots = {}
        for obj_id, obj_list in self.mcv_ann.items():
            annots[int(obj_id)] = obj_list.count(img_id)
        return annots


class HeadlessResnet2(nn.Module):
    def __init__(self, weights_path, embedding_size=64):
        super(HeadlessResnet2, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, embedding_size)
        if weights_path is not None:
            state_dict = torch.load(weights_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[6:]  # remove `model.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

    def inference(self, x):
        return self.model(x)

    def forward(self, x0, x1):
        output0 = self.model(x0)
        output1 = self.model(x1)
        return output0, output1


def train(model, dataloader, optimizer, criterion, cfg):
    model.train()
    # Iterate through the epochs
    for epoch in range(cfg["num_epochs"]):
        dataset_size = 0
        running_loss = 0
        # Iterate over batches
        with tqdm(dataloader, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch + 1}/{cfg['num_epochs']} train")
            for img0, img1, label in t_epoch:
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
                t_epoch.set_postfix({"loss": epoch_loss})
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")


def calculate_precision(y_true, y_pred, k_values):
    """
    Calculates the precision@k for a list of true labels (y_true) and predicted labels (y_pred)
    """
    correct_list = []
    for y_pred_i in y_pred:
        pred_correct = 0
        for k, v in y_true.items():
            if v == 1 and y_pred_i[k] == 1:
                pred_correct = 1
                break
        correct_list.append(pred_correct)

    # calculate retrieval precision
    precision = 0
    for i in range(k_values):
        precision += correct_list[i] / k_values
    return precision


def calculate_ap(y_true, y_pred):
    """
    Calculates the average precision (AP) for a list of true labels (y_true) and predicted labels (y_pred)
    """
    correct_list = []
    for y_pred_i in y_pred:
        pred_correct = 0
        for k, v in y_true.items():
            if v == 1 and y_pred_i[k] == 1:
                pred_correct = 1
                break
        correct_list.append(pred_correct)

    # calculate retrieval average precision
    ap = 0
    for i in range(len(correct_list)):
        ap += correct_list[i] / (i + 1)

    return ap


def plot_retrieval(query_img_path, path_to_images_retrieved_images, dists, query_gt, retrieved_gt):
    """
    Plots the retrieval results
    :param query_img_path: query image path
    :param path_to_images_retrieved_images: list of paths to the retrieved images
    :param dists: list of distances to the retrieved images
    :param query_gt: ground truth of the query image
    :param retrieved_gt: list of ground truths of the retrieved images
    """
    # Plot the query image and its ground truth
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    query_img = cv2.imread(query_img_path)[:, :, ::-1]
    ax.imshow(query_img)
    ax.set_title("Query image")
    ax.axis("off")
    query_items = query_gt.get('Id', [])
    if query_items:
        ax.text(0.5, -0.1, f"Query items: {query_items}", transform=ax.transAxes, ha="center")

    # Plot the retrieved images and their ground truth
    fig, ax = plt.subplots(1, len(path_to_images_retrieved_images), figsize=(20, 20))
    for i, path in enumerate(path_to_images_retrieved_images):
        img = cv2.imread(path)[:, :, ::-1]
        ax[i].imshow(img)
        ax[i].set_title(f"Distance: {dists[i]:.2f}")
        ax[i].axis("off")
        retrieved_items = retrieved_gt[i].get('Id', [])
        if retrieved_items:
            ax[i].text(0.5, -0.1, f"Retrieved items: {retrieved_items}", transform=ax[i].transAxes, ha="center")
    plt.show()


def evaluate_retrieval(model, dataloader, embedded_dataset, cfg):
    """
    Retrieves the k nearest images in the embedded space and evaluates the retrieval
    :param model: model to evaluate
    :param dataloader: dataloader of the dataset to evaluate
    :param embedded_dataset: dataset with the embeddings
    :param cfg: configuration dictionary
    :return: None
    """

    # Build a nearest neighbors model using the embedding dataset
    knn = NearestNeighbors(n_neighbors=int(cfg['k']), algorithm='ball_tree')
    embeddings, annots, paths = zip(*embedded_dataset)
    embeddings = np.array(embeddings)
    knn = knn.fit(embeddings)

    # Initialize lists for storing the ground truth annotations and retrieved images for each query
    gt_list = []
    mean_ap_list = []
    p_at_1_list = []
    p_at_5_list = []
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as batch:
            batch.set_description(f"Batches eval")
            for imgs, annotations, imgs_id in batch:
                # Send the images to CUDA
                embedded_imgs = imgs.to(cfg["device"])

                output = model.inference(embedded_imgs).cpu().numpy()

                q_dists, q_indices = knn.kneighbors(output)
                for i, (dists, indices) in enumerate(zip(q_dists, q_indices)):

                    # Get the ground truth annotations for the query image
                    y_true = {k: int(v[i].numpy()) for k, v in annotations.items()}
                    gt_list.append(y_true)

                    # Get the retrieved images and their annotations
                    pred = []
                    path_to_images = []
                    for j in indices:
                        pred.append(annots[j])
                        path_to_images.append(paths[j])

                    # Plot the query image and the retrieved images
                    if cfg["plot_retrieval"]:
                        plot_retrieval(imgs_id[i], path_to_images, dists, y_true, pred)

                    # Calculate AP, precision@1, and precision@5
                    ap = calculate_ap(y_true, pred)
                    precision_at_1 = calculate_precision(y_true, pred, k_values=1)
                    precision_at_5 = calculate_precision(y_true, pred, k_values=5)

                    mean_ap_list.append(ap)
                    p_at_1_list.append(precision_at_1)
                    p_at_5_list.append(precision_at_5)
                    batch.set_postfix({"AP@5": ap,
                                       "precision@1": precision_at_1,
                                       "precision@5": precision_at_5})

    mean_ap = sum(mean_ap_list) / len(mean_ap_list)
    mean_p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
    mean_p_at_5 = sum(p_at_5_list) / len(p_at_5_list)

    return mean_ap, mean_p_at_1, mean_p_at_5


def retrieve_dataset_embeddings(model: HeadlessResnet2, dataloader: DataLoader, out_path: str, cfg: Dict):
    embeddings = []

    if os.path.isfile(out_path):
        print("Loading embeddings from file")
        return np.load(out_path, allow_pickle=True)

    print("Retrieving dataset embeddings")
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as batch:
            batch.set_description(f"Batches eval")
            for imgs, annots, imgs_paths in batch:
                # Send the images to CUDA
                embedded_imgs = imgs.to(cfg["device"])

                output = model.inference(embedded_imgs).cpu().numpy()
                for i, embedding in enumerate(output):
                    annot = {k: int(v[i].numpy()) for k, v in annots.items()}
                    embeddings.append((embedding, annot, imgs_paths[i]))
    embedding = np.array(embeddings)
    # Save the array to disk
    np.save(out_path, embedding, allow_pickle=True)
    return embeddings


def main(cfg):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    if cfg["train"]:
        model = HeadlessResnet2(weights_path=None, embedding_size=cfg["embedder_size"]).to(cfg["device"])
        dataset = SiameseCOCODataset(cfg, "train", transform)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=cfg["batch_size"])
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = ContrastiveLoss()
        train(model, dataloader, optimizer, criterion, cfg)
    if cfg["evaluate"]:
        model = HeadlessResnet2(weights_path=cfg['eval_weights'], embedding_size=cfg["embedder_size"]).to(cfg["device"])

        db_dataset = SiameseCOCODataset(cfg, "database", transform, inference=True)
        db_dataloader = DataLoader(db_dataset, shuffle=False, num_workers=2, batch_size=cfg["batch_size"])
        embedded_dataset = retrieve_dataset_embeddings(model, db_dataloader, out_path=cfg["data_embeddings_path"],
                                                       cfg=cfg)

        val_dataset = SiameseCOCODataset(cfg, "val", transform, inference=True)
        val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=0, batch_size=cfg["batch_size"])
        m_ap, m_p_at_1, m_p_at_5 = evaluate_retrieval(model, val_dataloader, embedded_dataset, cfg)
        print(f"mAP@5: {m_ap:.4f} | mean precision@1: {m_p_at_1:.4f} | mean precision@5: {m_p_at_5:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_e.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
