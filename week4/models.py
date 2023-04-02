import torch.nn as nn
import torch
from torchvision import models
from collections import OrderedDict
from torchvision.models import ResNet18_Weights


class HeadlessResnet(nn.Module):
    def __init__(self, weights_path, embedding_size=64):
        super(HeadlessResnet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights)
        in_features = self.model.fc.in_features

        if weights_path is not None:
            self.model.fc = nn.Linear(in_features=in_features, out_features=8, bias=True)
            state_dict = torch.load(weights_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[6:]  # remove `model.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

        self.model.fc = nn.Identity()  # remove last layer
        # self.model.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x):
        output = self.model(x)
        return output


class Embedder(nn.Module):
    def __init__(self, output_size, embedding_size):
        super(Embedder, self).__init__()
        self.embedding = nn.Linear(output_size, embedding_size)

    def forward(self, x):
        output = self.embedding(x)
        return output
