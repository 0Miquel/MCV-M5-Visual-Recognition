import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class TrunkFasterRCNN(nn.Module):
    def __init__(self):
        super(TrunkFasterRCNN, self).__init__()
        self.features = None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # you can also hook layers inside the roi_heads
        self.layer_to_hook = 'roi_heads'
        for name, layer in self.model.named_modules():
            # if name == self.layer_to_hook:
            layer.register_forward_hook(self.save_features)

    def save_features(self, mod, inp, outp):
        self.features = outp

    def forward(self, anchor, positive, negative):
        _ = self.model(anchor)
        features_anchor = self.features
        _ = self.model(positive)
        features_positive = self.features
        _ = self.model(negative)
        features_negative = self.features

        return features_anchor, features_positive, features_negative
