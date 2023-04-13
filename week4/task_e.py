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


class TrunkFasterRCNN(nn.Module):
    def __init__(self):
        super(TrunkFasterRCNN, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        self.features = []
        # you can also hook layers inside the roi_heads
        self.layer_to_hook = 'roi_heads'
        for name, layer in self.model.named_modules():
            # if name == self.layer_to_hook:
            layer.register_forward_hook(self.save_features)

    def save_features(self, mod, inp, outp):
        self.features.append(outp)

    def forward(self, x):
        _ = self.model(x)
        return self.features[self.layer_to_hook]


def main():
    model = TrunkFasterRCNN()
    im = cv2.imread("../dataset/COCO/im001.jpg")[:, :, ::-1]
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    transformed_img = transform(image=im)["image"]
    model.eval()
    model(transformed_img[None, ...])


if __name__ == "__main__":
    main()
