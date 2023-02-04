import torch.nn as nn
from model.efficientnet import efficientnet0
from model.pointnet import PointNet

output_embedding_size = 128
point_net = PointNet(classes=output_embedding_size)
efficient_net = efficientnet0()(num_classes=output_embedding_size)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.pointnet = point_net
        self.efficientnet = efficient_net

    def forward_pointnet(self, x):
        # Forward pass
        output = self.pointnet.forward(x)
        return output

    def forward_efficientnet(self, x):
        output = self.efficientnet.forward(x)
        return output


    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_pointnet(input1)
        # forward pass of input 2
        output2 = self.forward_efficientnet(input2)
        return output1, output2
