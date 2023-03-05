import os
import numpy as np
import random
import torch
import pandas as pd

from src.utils.loading import load_point_cloud

from torch.utils.data import Dataset

from transformers import BertTokenizer


class TextPointCloudDataset(Dataset):
    def __init__(self, root_dir, pc_transform=None, stage="train"):
        self.root_dir = root_dir
        self.stage = stage

        self.text_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.pc_transforms = pc_transform

        text_queries_folder_path = os.path.join(root_dir, "text_queries")

        ground_truth_path = os.path.join(
            text_queries_folder_path, "TextQuery_GT_Train.csv"
        )
        validation_ground_truth_path = os.path.join(
            text_queries_folder_path, "TextQuery_GT_Validation.csv"
        )
        text_queries_path = os.path.join(
            text_queries_folder_path, "TextQuery_Train.csv"
        )
        model_id_path = os.path.join(text_queries_folder_path, "ModelID.csv")

        self.point_cloud_path = os.path.join(root_dir, "PC_OBJ")

        self.ground_truth_csv = pd.read_csv(ground_truth_path, sep=";")
        self.validation_ground_truth_csv = pd.read_csv(
            validation_ground_truth_path, sep=";"
        )
        self.text_queries_csv = pd.read_csv(text_queries_path, sep=";")

        self.point_cloud_ids_list = pd.read_csv(model_id_path)["ID"].to_list()
        self.text_queries_mapping = dict(self.text_queries_csv.values)

        self.text_queries_list = self.ground_truth_csv["Text Query ID"].to_list()
        self.point_cloud_list = self.ground_truth_csv["Model ID"].to_list()

        self.validation_text_queries_list = self.validation_ground_truth_csv[
            "Text Query ID"
        ].to_list()
        self.validation_point_cloud_list = self.validation_ground_truth_csv[
            "Model ID"
        ].to_list()

        self.queries_model_mapping = dict()
        for i, text in enumerate(self.text_queries_list):
            if text not in self.queries_model_mapping:
                self.queries_model_mapping[text] = set()
            self.queries_model_mapping[text].add(self.point_cloud_list[i])

    def _preprocess_pc(self, filename):
        pointcloud = load_point_cloud(filename)
        if self.pc_transforms:
            pointcloud = self.pc_transforms(pointcloud)
        return pointcloud

    def _preprocess_batch_text(self, batch):
        return self.text_tokenizer.batch_encode_plus(
            batch, padding="longest", return_tensors="pt"
        )

    def __len__(self):
        if self.stage == "train":
            return len(self.text_queries_list)
        else:
            return len(self.validation_text_queries_list)

    def __getitem__(self, idx):
        if self.stage == "train":
            text_queries_id = self.text_queries_list[idx]
            true_point_cloud_id = false_point_cloud_id = self.point_cloud_list[idx]
        else:
            text_queries_id = self.validation_text_queries_list[idx]
            true_point_cloud_id = (
                false_point_cloud_id
            ) = self.validation_point_cloud_list[idx]

        while false_point_cloud_id in self.queries_model_mapping[text_queries_id]:
            false_point_cloud_id = self.point_cloud_ids_list[
                random.randint(0, len(self.point_cloud_ids_list) - 1)
            ]
            # false_point_cloud_id = 'f770b7a17bed6938'

        true_pc_path = os.path.join(self.point_cloud_path, f"{true_point_cloud_id}.obj")
        false_pc_path = os.path.join(
            self.point_cloud_path, f"{false_point_cloud_id}.obj"
        )

        true_point_cloud = self._preprocess_pc(true_pc_path)
        false_point_cloud = self._preprocess_pc(false_pc_path)

        text_sample = self.text_queries_mapping[text_queries_id]

        return {
            "true_point_cloud": true_point_cloud,
            "false_point_cloud": false_point_cloud,
            "text_query": text_sample,
            "true_point_cloud_id": true_point_cloud_id,
            "text_query_id": text_queries_id,
        }

    def collate_fn(self, batch):
        batch_as_dict = {
            "true_point_clouds": torch.stack([x["true_point_cloud"] for x in batch])
            .float()
            .transpose(1, 2),
            "false_point_clouds": torch.stack([x["false_point_cloud"] for x in batch])
            .float()
            .transpose(1, 2),
            "text_queries": self._preprocess_batch_text(
                [x["text_query"] for x in batch]
            ),
            "true_point_cloud_ids": [x["true_point_cloud_id"] for x in batch],
            "text_query_ids": [x["text_query_id"] for x in batch],
        }

        return batch_as_dict
