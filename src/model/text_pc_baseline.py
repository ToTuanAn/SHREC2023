import torch
import random
import math
import os
import torch.nn as nn

from src.extractor.base_pc_extractor import PointNet
from src.extractor.bert_extractor import LangExtractor

class TextPointCloudNetwork(nn.Module):
    def __init__(self, num_classes=128):
        #TODO: change num_classes to config.feature_embedding_sizes
        super().__init__()
        self.pointnet = PointNet()
        self.lang_extractor = LangExtractor('bert-base-uncased', True)
        self.pointnet_linear = nn.Linear(self.pointnet.feature_dim, num_classes)
        self.lang_extractor_linear = nn.Linear(self.lang_extractor.feature_dim, num_classes)

    def forward_pointnet(self, x):
        # Forward pass
        output, _, _ = self.pointnet.forward(x)
        output = self.pointnet_linear(output)
        return output

    def forward_lang_extractor(self, x):
        output = self.lang_extractor.forward(x)
        output = self.lang_extractor_linear(output)
        return output

    def forward(self, batch):
        true_point_cloud, false_point_cloud, text_queries = batch['true_point_cloud'], batch['false_point_cloud'], batch['text_queries']
        true_pointnet_embedding_feature = self.forward_pointnet(true_point_cloud)
        false_pointnet_embedding_feature = self.forward_pointnet(false_point_cloud)
        text_queries_embedding_feature = self.forward_lang_extractor(text_queries)

        return true_pointnet_embedding_feature, false_pointnet_embedding_feature, text_queries_embedding_feature

    def compute_loss(self, batch, **kwargs):
        # TODO: write method to compute loss
        # diff = x0 - x1
        # dist_sq = torch.sum(torch.pow(diff, 2), 1)
        # dist = torch.sqrt(dist_sq)
        #
        # mdist = self.margin - dist
        # dist = torch.clamp(mdist, min=0.0)
        # loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        # loss = torch.sum(loss) / 2.0 / x0.size()[0]
        pass