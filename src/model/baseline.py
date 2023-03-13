from pytorch_metric_learning.losses import CrossBatchMemory, NTXentLoss
from src.model.abstract import MLP, AbstractModel
from src.extractor.base_pc_extractor import PointNetExtractor
from src.extractor.bert_extractor import LangExtractor
import torch.nn as nn


class BaselineModel(AbstractModel):
    def init_model(self):
        self.pc_extractor = PointNetExtractor()
        self.lang_extractor = LangExtractor(pretrained="bert-base-uncased", freeze=True)
        self.embed_dim = self.cfg["model"]["params"]["embed_dim"]

        self.lang_encoder = MLP(self.lang_extractor, self.embed_dim)
        self.pc_encoder = MLP(self.pc_extractor, self.embed_dim)

    def forward(self, batch):
        pc_embedding_feats = self.pc_encoder.forward(batch["point_clouds"])
        query_embedding_feats = self.lang_encoder.forward(batch["queries"])

        return {
            "pc_embedding_feats": pc_embedding_feats,
            "query_embedding_feats": query_embedding_feats,
        }

    def compute_loss(self, batch, **kwargs):
        contra_loss = NTXentLoss()
        cbm_query = CrossBatchMemory(contra_loss, latent_dim, 128)
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

        return triplet_loss(
            anchor=kwargs["text_query_embedding_feats"],
            positive=kwargs["pc_embedding_feats"],
            negative=kwargs["pc_embedding_feats"],
        )
