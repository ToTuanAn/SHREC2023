from pytorch_metric_learning.losses import CrossBatchMemory, NTXentLoss
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch

from src.model.abstract import MLP, AbstractModel
from src.extractor.base_pc_extractor import PointNetExtractor
from src.extractor.bert_extractor import LangExtractor


class BaselineModel(AbstractModel):
    def init_model(self):
        self.pc_extractor = PointNetExtractor()
        self.lang_extractor = LangExtractor(pretrained="bert-base-uncased", freeze=True)
        self.embed_dim = self.cfg["model"]["embed_dim"]

        self.lang_encoder = MLP(self.lang_extractor, self.embed_dim)
        self.pc_encoder = MLP(self.pc_extractor, self.embed_dim)

        self.constrastive_loss = NTXentLoss()
        self.xbm = CrossBatchMemory(
            loss=self.constrastive_loss,
            embedding_size=self.embed_dim,
            memory_size=self.cfg["model"]["xbm"]["memory_size"],
        )

    def forward(self, batch):
        pc_embedding_feats = self.pc_encoder.forward(batch["point_clouds"])
        query_embedding_feats = self.lang_encoder.forward(batch["queries"])

        return {
            "pc_embedding_feats": pc_embedding_feats,
            "query_embedding_feats": query_embedding_feats,
        }

    def compute_loss(self, forwarded_batch, input_batch):
        """
        Concatenatae point cloud embedding feature and query embedding feature
        to calculate pair-based loss (here we use InfoNCE loss).
        Label are generated from query_ids (here we consider each query as a "class").
        First we train the model with InfoNCE loss. After certain step, we apply
        Cross Batch Memory method in addiontional with InfoNCE to increase hard mining ability.

        Args:
            forwarded_batch: output of `forward` method
            input_batch: input of batch method

        Returns:
            loss: computed loss
        """
        emb = torch.cat(
            [
                forwarded_batch["pc_embedding_feats"],
                forwarded_batch["query_embedding_feats"],
            ]
        )  # (batch_size * 2, embed_dim)
        emb_len = emb.shape[0]

        # label is categoricalized id of queries (but repeated 2 time since we concated the pc and query)
        labels = torch.tensor(
            LabelEncoder().fit_transform(input_batch["query_ids"]), dtype=torch.int
        ).repeat(
            2
        )  # (batch_size * 2)

        if self.current_epoch >= self.cfg["model"]["xbm"]["enable_epoch"]:
            loss = self.xbm(emb, labels)
        else:
            loss = self.constrastive_loss(emb, labels)
        return loss
