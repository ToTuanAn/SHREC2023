from src.model.abstract import AbstractModel
from src.extractor.img_extractor import EfficientNetExtractor
from sec.extractor.pc_extractor import PointNet
from . import MODEL_REGISTRY


@MODEL_REGISTRY.register
class SiameseNetwork(AbstractModel):
    def init_model(self):
        self.pointnet = PointNet(classes=self.cfg.model.output_embedding_size)
        self.efficientnet = EfficientNetExtractor(version=0.1)

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
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        pass
