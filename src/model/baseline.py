from src.model.abstract import AbstractModel
import torch.nn as nn
from torchvision import transforms, utils
from src.model.efficientnet import efficientnet0
from src.model.pointnet import PointNet
from . import MODEL_REGISTRY
from src.loader.loader import SiameseNetworkDataset
from ..loader.pointnetloader import Normalize, ToTensor

MODEL_REGISTRY.register()
root_dir = '../../dataset/'

pc_transforms = transforms.Compose([
    Normalize(),
    ToTensor()
])

img_transforms = transforms.Comnpose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = SiameseNetworkDataset(root_dir, pc_transform=pc_transforms, img_transform=img_transforms)


class SiameseNetwork(AbstractModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_model(self):
        self.pointnet = PointNet(classes=self.cfg.model.output_embedding_size)
        self.efficientnet = efficientnet0(num_classes=self.cfg.model.output_embedding_size)

    def forward_pointnet(self, x):
        # Forward pass
        output = self.pointnet.forward(x)
        return output

    def forward_efficientnet(self, x):
        output = self.efficientnet.forward(x)
        return output

    def forward(self, batch):
        # TODO: write forward method
        pass

    def compute_loss(self, batch, **kwargs):
        # TODO: write method to compute loss
        pass