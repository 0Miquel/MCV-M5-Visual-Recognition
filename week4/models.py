import torch.nn as nn
import torch
from torchvision import models
from collections import OrderedDict
from torchvision.models import ResNet18_Weights


class HeadlessResnet(nn.Module):
    def __init__(self, weights_path, inference=False):
        super(HeadlessResnet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights)
        in_features = self.model.fc.in_features

        if weights_path is not None:
            if not inference:
                self.model.fc = nn.Linear(in_features=in_features, out_features=8, bias=True)
            else:
                self.model.fc = nn.Identity()
            state_dict = torch.load(weights_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[6:]  # remove `model.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

        if not inference:
            self.model.fc = nn.Identity()  # remove last layer

    def forward(self, x):
        output = self.model(x)
        return output


class Embedder(nn.Module):
    def __init__(self, output_size, embedding_size, weights_path=None):
        super(Embedder, self).__init__()
        self.embedding = nn.Linear(output_size, embedding_size)

        if weights_path is not None:
            state_dict = torch.load(weights_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[10:]  # remove `model.`
                new_state_dict[name] = v
            self.embedding.load_state_dict(new_state_dict)

    def forward(self, x):
        output = self.embedding(x)
        return output


class Fusion(nn.Module):
    def __init__(self, trunk, embedder):
        super(Fusion, self).__init__()
        self.trunk = trunk
        self.embedder = embedder

    def forward(self, x):
        x = self.trunk(x)
        output = self.embedder(x)
        return output
