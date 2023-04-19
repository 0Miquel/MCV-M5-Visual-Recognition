from PIL import Image
from torch.utils.data import Dataset

import json
import random


class TripletIm2Text(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

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
        return len(self.images)

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

        return image, positive_caption, negative_caption


class TripletText2Im(Dataset):
    def __init__(self, ann_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)

        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']

        # Create dictionary where key is image id and value is a list of the captions related to that image
        self.caption2img = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.caption2img:
                self.caption2img[img_id] = [i]
            else:
                self.caption2img[img_id].append(i)

    def __len__(self):
        return len(self.annotations_an)

    def __getitem__(self, index):
        anchor_caption = self.annotations_an[index]

        positive_img_id = anchor_caption['image_id']
        positive_img_path = self.images[positive_img_id]['file_name']

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

        return image, positive_caption, negative_caption
