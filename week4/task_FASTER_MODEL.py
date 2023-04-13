import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class TrunkFasterRCNN(nn.Module):
    def __init__(self):
        super(TrunkFasterRCNN, self).__init__()
        self.features = None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.back = self.model.backbone

        # you can also hook layers inside the roi_heads
        # self.layer_to_hook = 'roi_heads.box_head'
        # for name, layer in self.model.named_modules():
        #     if name == self.layer_to_hook:
        #         layer.register_forward_hook(self.save_features)

    def save_features(self, mod, inp, outp):
        self.features = outp

    def forward(self, x0, x1):
        features_x0 = self.back(x0)['pool'].view(8, -1)
        features_x1 = self.back(x1)['pool'].view(8, -1)

        return features_x0, features_x1
