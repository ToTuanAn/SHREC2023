from src.model.abstract import AbstractModel
from src.extractor.base_pc_extractor import PointNet
from src.extractor.bert_extractor import LangExtractor
import torch.nn as nn


class BaselineModel(AbstractModel):
    def init_model(self):
        self.pointnet = PointNet()
        self.lang_extractor = LangExtractor("bert-base-uncased", True)

        self.lang_linear = nn.Sequential(
            nn.Linear(self.lang_extractor.feature_dim, self.lang_extractor.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.lang_extractor.feature_dim // 2, self.lang_extractor.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.lang_extractor.feature_dim // 4, self.cfg["model"]["params"]["embed_dim"]),
        )
        self.pc_linear = nn.Linear(
            self.pointnet.feature_dim, self.cfg["model"]["params"]["embed_dim"]
        )

    def forward_pointnet(self, batch_pc):
        output, _, _ = self.pointnet.forward(batch_pc)
        output = self.pc_linear(output)
        return output

    def forward_lang_extractor(self, batch_lang):
        output = self.lang_extractor.forward(batch_lang)
        output = self.lang_linear(output)
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
            anchor=kwargs["text_query_embedding_feats"],
            positive=kwargs["true_pc_embedding_feats"],
            negative=kwargs["false_pc_embedding_feats"],
        )
