# create an encoder using resnet18 or resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(
        self, encoder_type="resnet18", pretrained=True, weights=None, cfg=None
    ):
        super(ImageEncoder, self).__init__()
        self.cfg = cfg
        if encoder_type == "resnet18":
            self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder.fc = nn.Identity()
        elif encoder_type == "resnet50":
            self.encoder = models.resnet50(pretrained=pretrained)
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError("Unsupported encoder type: {}".format(encoder_type))

        if weights:
            self.encoder.load_state_dict(torch.load(weights))

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)
