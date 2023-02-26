from src.model.abstract import AbstractModel
from src.extractor.base_pc_extractor import PointNet
from src.extractor.bert_extractor import LangExtractor
from . import MODEL_REGISTRY
import torch.nn as nn


@MODEL_REGISTRY.register()
class BaselineModel(AbstractModel):
    def init_model(self):
        self.pointnet = PointNet()
        self.lang_extractor = LangExtractor("bert-base-uncased", True)
        self.pointnet_linear = nn.Linear(
            self.pointnet.feature_dim, self.cfg["model"]["embed-size"]
        )
        self.lang_extractor_linear = nn.Linear(
            self.lang_extractor.feature_dim, self.cfg["model"]["embed-size"]
        )

    def forward_pointnet(self, batch_pc):
        # Forward pass
        output, _, _ = self.pointnet.forward(batch_pc)
        output = self.pointnet_linear(output)
        return output

    def forward_lang_extractor(self, batch_lang):
        output = self.lang_extractor.forward(batch_lang)
        output = self.lang_extractor_linear(output)
        return output

    def forward(self, batch):
        true_pc_embedding_feats = self.forward_pointnet(batch["true_point_clouds"])
        false_pc_embedding_feats = self.forward_pointnet(batch["false_point_clouds"])
        text_query_embedding_feats = self.forward_lang_extractor(batch["text_queries"])

        return {
            "true_pc_embedding_feats": true_pc_embedding_feats,
            "false_pc_embedding_feats": false_pc_embedding_feats,
            "text_query_embedding_feats": text_query_embedding_feats,
        }

    def compute_loss(self, batch, **kwargs):
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        return triplet_loss(
            anchor=batch["text_query_embedding_feats"],
            positive=batch["true_pc_embedding_feats"],
            negative=batch["false_pc_embedding_feats"],
        )
