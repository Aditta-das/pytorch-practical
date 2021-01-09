import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss Functions


# Types of VGG
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

# model
class VGG_net(nn.Module):
	def __init__(self, in_channels=3, num_classes=1000):
		super(VGG_net, self).__init__()
		self.in_channels = in_channels
		self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

	def forward(self, x):
		pass

	def create_conv_layers(self, architecture):
		pass