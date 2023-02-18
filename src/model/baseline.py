from src.model.abstract import AbstractModel
from src.model.efficientnet import efficientnet0
from src.model.pointnet import PointNet
from . import MODEL_REGISTRY

root_dir = "../../dataset/"

MODEL_REGISTRY.register()


class SiameseNetwork(AbstractModel):
    def init_model(self):
        self.pointnet = PointNet(classes=self.cfg.model.output_embedding_size)
        self.efficientnet = efficientnet0(
            num_classes=self.cfg.model.output_embedding_size
        )

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
