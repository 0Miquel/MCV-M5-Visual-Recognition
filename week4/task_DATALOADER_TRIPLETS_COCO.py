"""
This code has been written following the tutorial
https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/"""
import argparse
import json
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from week4.coco_classes import coco_id_class_mapping
from week4.i_o import load_yaml_config


def image_shares_classes(ann_img1: Dict, ann_img2: Dict) -> bool:
    """
    Checks if the two images share at least one class
    :param ann_img1: annotations of the first image
    :param ann_img2: annotations of the second image
    :return: True if the two images share at least one class, False otherwise
    """
    share_ids = False
    ids_iter = iter(ann_img1.keys())
    while not share_ids:
        try:
            item = next(ids_iter)
            if ann_img1[item] > 0 and ann_img2[item] > 0:
                share_ids = True
        except StopIteration:
            break
    return share_ids


class SiameseNetworkDataset(Dataset):
    def __init__(self, msc_ann_path: str, subset: str = 'train', transform=None, cfg: Dict = None):
        self.msc_ann_path = msc_ann_path
        self.transform = transform
        self.imgs_path = cfg[f'{subset}_path'] + f"COCO_{subset}2014_"
        self.mcv_ann = json.load(open(self.msc_ann_path))[subset]
        self.classes = list(self.mcv_ann.keys())

    def __getitem__(self, index):
        class_idx = np.random.randint(len(self.classes))
        selected_class = self.classes[class_idx]
        selected_imgs_ids = np.random.choice(self.mcv_ann[selected_class], size=2, replace=False)
        selected_imgs_annots = [self._load_annot(img_id) for img_id in selected_imgs_ids]
        label = 0  # same class

        if np.random.rand() < 0.5:
            different_class = np.random.choice(list(set(self.classes) - {selected_class}))
            label = 1  # different class
            while image_shares_classes(selected_imgs_annots[0], selected_imgs_annots[1]):
                selected_imgs_ids[1] = np.random.choice(self.mcv_ann[different_class])
                selected_imgs_annots[1] = self._load_annot(selected_imgs_ids[1])

        image_pair = []
        for image_id in selected_imgs_ids:
            path_name = self.imgs_path + "{:012d}".format(image_id) + ".jpg"
            image = Image.open(path_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            image_pair.append(image)

        tuple_and_label = ((image_pair[0], image_pair[1]), tuple(selected_imgs_annots), label, selected_class)
        return tuple_and_label

    def __len__(self):
        with open(self.msc_ann_path) as f:
            msc_ann = json.load(f)

        num_pairs = sum(len(image_paths) for image_paths in msc_ann['train'].values()) * 2

        return num_pairs

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

        triplet = (anchor, positive, negative)
        return triplet

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


def visualize_duplet(
        anchor: np.ndarray,
        pick: np.ndarray,
        annot_anchor: Dict,
        annot_pick: Dict,
        similar: bool,
        query_id: str,
        i: int,
        category_mapping: Dict[int, str]
):
    appearing_ids_anchor = [k for k, v in annot_anchor.items() if v[i] > 0]
    appearing_ids_pick = [k for k, v in annot_pick.items() if v[i] > 0]

    appearing_names_anchor = [category_mapping[id] for id in appearing_ids_anchor]
    appearing_names_pick = [category_mapping[id] for id in appearing_ids_pick]

    text_anchor = f"Anchor: {', '.join(appearing_names_anchor)}"
    similarity = "Similar" if similar else "Dissimilar"
    text_pick = f"{similarity}: {', '.join(appearing_names_pick)}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    fig.suptitle(f"Query: {category_mapping[int(query_id)]}")

    ax1.imshow(np.transpose(anchor, (1, 2, 0)))
    ax1.set_title(text_anchor)
    ax1.axis("off")

    ax2.imshow(np.transpose(pick, (1, 2, 0)))
    ax2.set_title(text_pick)
    ax2.axis("off")


def show_siamese_batch(siamese_batch: torch.Tensor, batch_size: int = 8):
    siamese_batch = siamese_batch
    anchor_batch, pick_batch = siamese_batch[0]
    annot_anchor_batch, annot_pick_batch = siamese_batch[1]
    similarity_batch = siamese_batch[2].numpy()
    selected_id_batch = siamese_batch[3]
    class_mapping = coco_id_class_mapping  # Load category mapping

    for i in range(batch_size):
        anchor = anchor_batch[i].numpy()
        pick = pick_batch[i].numpy()

        annot_anchor = annot_anchor_batch
        annot_pick = annot_pick_batch
        query_id = selected_id_batch[i]

        visualize_duplet(anchor, pick, annot_anchor, annot_pick, not similarity_batch[i], query_id, i, class_mapping)
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
    # siamese_dataset = TripletNetworkDataset(msc_ann_path=cfg["msc_ann"],
    #                                         transform=transformation,
    #                                         cfg=cfg)
    siamese_dataset = SiameseNetworkDataset(msc_ann_path=cfg["msc_ann"],
                                            transform=transformation,
                                            cfg=cfg)

    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=2,
                                batch_size=8)

    example_batch = next(iter(vis_dataloader))

    # plot all the images
    show_siamese_batch(example_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_e.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = load_yaml_config(config_path)

    main(config)
