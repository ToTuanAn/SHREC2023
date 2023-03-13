import os
import numpy as np
import random
import torch
import pandas as pd

from src.utils.loading import load_point_cloud

from torch.utils.data import Dataset

from transformers import BertTokenizer


class InferenceDataset(Dataset):
    def __init__(self, root_dir, pc_transform=None, stage="test"):
        self.root_dir = root_dir
        self.stage = stage

        self.text_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.pc_transforms = pc_transform

        text_queries_folder_path = os.path.join(root_dir, "text_queries")

        text_queries_path = os.path.join(text_queries_folder_path, "TextQuery_Test.csv")

        model_id_path = os.path.join(text_queries_folder_path, "ModelID.csv")

        self.point_cloud_path = os.path.join(root_dir, "PC_OBJ")

        self.text_queries_csv = pd.read_csv(text_queries_path, sep=";")

        self.point_cloud_ids_list = pd.read_csv(model_id_path)["ID"].to_list()
        self.text_ids_list = pd.read_csv(text_queries_path, sep=";")["ID"].to_list()

        self.text_queries_mapping = dict(self.text_queries_csv.values)

        self.inference_test = []

        for point_cloud_id in self.point_cloud_ids_list:
            for text_id in self.text_ids_list:
                self.inference_test.append((point_cloud_id, text_id))

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
        return len(self.inference_test)

    def __getitem__(self, idx):
        point_cloud_id, text_id = self.inference_test[idx]

        # print(text_queries_id, true_point_cloud_id, false_point_cloud_id)
        text_sample = self.text_queries_mapping[text_id]
        text_input_ids, text_attention_mask = self._preprocess_text(text_sample)

        pc_path = os.path.join(self.point_cloud_path, f"{point_cloud_id}.obj")

        true_point_cloud = self._preprocess_pc(pc_path)

        return {
            "point_cloud_id": point_cloud_id,
            "point_cloud": true_point_cloud,
            "text_id": text_id,
            "input_ids": text_input_ids,
            "attention_mask": text_attention_mask,
        }

    def collate_fn(self, batch):
        batch_as_dict = {
            "true_point_clouds": torch.stack([x["true_point_cloud"] for x in batch])
            .float()
            .transpose(1, 2),
            "false_point_clouds": torch.stack([x["false_point_cloud"] for x in batch])
            .float()
            .transpose(1, 2),
            "text_queries": {
                "input_ids": torch.stack([x["input_ids"] for x in batch]),
                "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            },
        }

        return batch_as_dict
