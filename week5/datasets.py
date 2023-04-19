import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
import os
import random
from PIL import Image
import torchvision.datasets as datasets
import nltk
nltk.download('punkt')


class CocoCaptionsWithNegative(datasets.CocoCaptions):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root=root, annFile=annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Select a random annotation to use as the positive example
        pos_ann = random.choice(annotations)
        pos_caption = pos_ann['caption']

        # Select another random annotation from the same image to use as the negative example
        neg_anns = [ann for ann in annotations if ann != pos_ann]
        neg_ann = random.choice(neg_anns)
        neg_caption = neg_ann['caption']

        # Load the image
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, pos_caption, neg_caption
