import json
import os
import random
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from week5.src.models import TripletNetIm2Text, TripletNetText2Im


class TripletIm2Text(Dataset):
    def __init__(self, ann_file, img_dir, transform=None, evaluation=False):
        self.img_dir = img_dir
        self.transform = transform
        self.evaluation = evaluation

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)

        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']

        # Create dictionary where key is image id and value is a list of the captions related to that image
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.img2ann:
                self.img2ann[img_id] = [i]
            else:
                self.img2ann[img_id].append(i)

    def __len__(self):
        return int(len(self.images) * 0.1)

    def __getitem__(self, index):
        img_path = self.img_dir + '/' + self.images[index]['file_name']
        img_id = self.images[index]['id']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Choose randomly one captions for the image
        idx_pos = random.choice(self.img2ann[img_id])
        assert self.annotations_an[idx_pos]['image_id'] == img_id
        positive_caption_id = self.annotations_an[idx_pos]['id']
        positive_caption = self.annotations_an[idx_pos]['caption']

        # Choose randomly one caption that is not the same as the positive caption
        negative_caption_id = positive_caption_id
        while negative_caption_id == positive_caption_id:
            neg_ann = random.choice(self.annotations_an)
            if neg_ann['image_id'] == img_id:
                continue
            else:
                negative_caption_id = neg_ann['id']

        negative_caption = neg_ann['caption']

        # Lower case
        positive_caption = positive_caption.lower()
        negative_caption = negative_caption.lower()
        if not self.evaluation:
            return image, positive_caption, negative_caption
        else:
            captions = [self.annotations_an[i]['id'] for i in self.img2ann[img_id][:5]]
            return image, captions


class TripletText2Im(Dataset):
    def __init__(self, ann_file, img_dir, phase, transform=None, evaluation=False):
        self.phase = phase
        self.img_dir = img_dir
        self.transform = transform
        self.evaluation = evaluation

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)

        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']

        # Create dictionary where key is image id and value is a list of the captions related to that image
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.img2ann:
                self.img2ann[img_id] = [i]
            else:
                self.img2ann[img_id].append(i)

    def __len__(self):
        return int(len(self.annotations_an) * 0.1)

    def __getitem__(self, index):
        anchor = self.annotations_an[index]
        anchor_caption = anchor['caption']
        positive_img_id = anchor['image_id']
        # create positive image path with positiva img id, "COCO_train2014_" and
        # then positive img id up to 12 digits with 0s, ending in jpg
        positive_img_path = self.img_dir + '/' + 'COCO_' + self.phase + '2014_' + str(positive_img_id).zfill(
            12) + '.jpg'

        positive_img = Image.open(positive_img_path).convert('RGB')
        if self.transform is not None:
            positive_img = self.transform(positive_img)

        # Choose randomly one caption that is not the same as the positive caption
        # get a random key from caption2img dict that is not positive_img_id
        negative_img_id = positive_img_id
        while negative_img_id == positive_img_id:
            negative_img_id = random.choice(list(self.img2ann.keys()))
        negative_img_path = self.img_dir + '/' + 'COCO_' + self.phase + '2014_' + str(negative_img_id).zfill(
            12) + '.jpg'

        negative_img = Image.open(negative_img_path).convert('RGB')
        if self.transform is not None:
            negative_img = self.transform(negative_img)

        # Lower case
        anchor_caption = anchor_caption.lower()

        if not self.evaluation:
            return anchor_caption, positive_img, negative_img
        else:
            return anchor_caption, positive_img_id


def create_caption_db(model: TripletNetIm2Text, captions: Dict, out_path: str, device: str = 'cpu'):
    if os.path.isfile(out_path):
        print("Loading embeddings from file")
        return np.load(out_path, allow_pickle=True)

    print("Retrieving dataset embeddings")

    caption_embeddings = []
    for cap_id, caption in tqdm(captions.items()):
        if device == 'cpu':
            caption_embedding = np.array(model.get_embedding_text([caption])[0])
        else:
            caption_embedding = model.get_embedding_text([caption])[0].cpu().detach().numpy()
        caption_embeddings.append((caption_embedding, cap_id))

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    caption_embeddings = np.array(caption_embeddings)
    # Save the array to disk
    np.save(out_path, caption_embeddings, allow_pickle=True)

    return caption_embeddings


def create_image_db(model: TripletNetText2Im, im_dict: Dict, im_dir: str, out_path: str, device: str = 'cpu', transform=None):
    if os.path.isfile(out_path):
        print("Loading embeddings from file")
        return np.load(out_path, allow_pickle=True)

    print("Retrieving dataset embeddings")

    image_embeddings = []
    for im_id, img_path in tqdm(im_dict.items()):
        img = Image.open(f'{im_dir}/{img_path}').convert('RGB')
        if transform is not None:
            img = transform(img)
        img = img.expand(1, -1, -1, -1)
        img.to(device)
        if device == 'cpu':
            img_embedding = model.get_embedding_image(img)[0].detach().numpy()
        else:
            img_embedding = model.get_embedding_image(img)[0].cpu().detach().numpy()
        image_embeddings.append((img_embedding, im_id))

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    caption_embeddings = np.array(image_embeddings)
    # Save the array to disk
    np.save(out_path, caption_embeddings, allow_pickle=True)

    return caption_embeddings
