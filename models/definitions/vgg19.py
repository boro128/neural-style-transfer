import torch.nn as nn

from torchvision import models
from collections import namedtuple


class Vgg19(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        pretrained_features = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.layers_names = ['conv1_1', 'conv2_1',
                             'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        self.content_feature_maps_idx = 4
        self.style_feature_maps_indices = list(range(len(self.layers_names)))
        self.style_feature_maps_indices.remove(self.content_feature_maps_idx)

        # replace MaxPool layers with AvgPool as suggested in the paper
        for i, layer in pretrained_features.named_children():
            if isinstance(layer, nn.MaxPool2d):
                pretrained_features[int(i)] = nn.AvgPool2d(kernel_size=layer.kernel_size,
                                                           stride=layer.stride,
                                                           padding=layer.padding,
                                                           ceil_mode=layer.ceil_mode)

        self.block1 = nn.Sequential()  # outputs conv1_1
        self.block2 = nn.Sequential()  # outputs conv2_1
        self.block3 = nn.Sequential()  # outputs conv3_1
        self.block4 = nn.Sequential()  # outputs conv4_1
        self.block5 = nn.Sequential()  # outputs conv4_2
        self.block6 = nn.Sequential()  # outputs conv5_1

        self.block1.append(pretrained_features[0])
        for i in range(1, 6):
            self.block2.append(pretrained_features[i])
        for i in range(6, 11):
            self.block3.append(pretrained_features[i])
        for i in range(11, 20):
            self.block4.append(pretrained_features[i])
        for i in range(20, 22):
            self.block5.append(pretrained_features[i])
        for i in range(22, 29):
            self.block6.append(pretrained_features[i])

        # prevent the model from updating the weights
        # in this task the only learnable parameter is the resulting image
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        conv1_1 = self.block1(x)
        conv2_1 = self.block(conv1_1)
        conv3_1 = self.block(conv2_1)
        conv4_1 = self.block(conv3_1)
        conv4_2 = self.block(conv4_1)
        conv5_1 = self.block(conv4_2)

        outputs = namedtuple('Outputs', self.layers_names)
        return outputs(conv1_1, conv2_1, conv3_1, conv4_1, conv4_2, conv5_1)
