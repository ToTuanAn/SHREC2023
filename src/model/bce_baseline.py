import torch
import numpy as np

from src.model.abstract import AbstractModel
from src.extractor.base_pc_extractor import PointNet
from torch.autograd import Variable
from src.extractor.bert_extractor import LangExtractor
import torch.nn as nn


class BCEPointCloudTextModel(AbstractModel):
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

        self.fc = nn.Sequential(
            nn.Linear(self.cfg["model"]["params"]['embed_dim'] * 2, 1),
            nn.Sigmoid()
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

        positive_sample = torch.cat((true_pc_embedding_feats, text_query_embedding_feats), 1)
        negative_sample = torch.cat((false_pc_embedding_feats, text_query_embedding_feats), 1)

        positive_prob = self.fc(positive_sample)
        negative_prob = self.fc(negative_sample)



        return {
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
        }

    def compute_loss(self, batch, **kwargs):
        bce_loss = nn.BCEWithLogitsLoss()
        true_labels = torch.full(size=kwargs["positive_prob"].shape, fill_value=1).float().cuda()

        false_labels = torch.full(size=kwargs["negative_prob"].shape, fill_value=0).float().cuda()

        return ( bce_loss(kwargs["positive_prob"], true_labels) + bce_loss(kwargs["negative_prob"], false_labels) ) / 2
