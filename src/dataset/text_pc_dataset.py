import os
import numpy as np
import random
import torch
import pandas as pd

from src.preprocessing import load_point_cloud

random.seed = 42

from torch.utils.data import Dataset
from torchvision.io import read_image

from src.dataset.utils import read_pc

from transformers import BertTokenizer


class TextPointCloudDataset(Dataset):
    def __init__(self, root_dir, pc_transform=None):
        self.root_dir = root_dir

        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                            do_lower_case=True)
        self.pc_transforms = pc_transform

        text_queries_folder_path = os.path.join(root_dir, "text_queries")
        ground_truth_path = os.path.join(text_queries_folder_path, "TextQuery_GT_Train.csv")
        text_queries_path = os.path.join(text_queries_folder_path, "TextQuery_Train.csv")

        self.point_cloud_path = os.path.join(root_dir, "PC_OBJ")

        self.ground_truth_csv = pd.read_csv(ground_truth_path, sep=";")
        self.text_queries_csv = pd.read_csv(text_queries_path, sep=";")
        self.text_queries_mapping = dict(self.text_queries_csv.values)

        print(self.text_queries_mapping)

        self.text_queries_list = self.ground_truth_csv['Text Query ID'].to_list()
        self.point_cloud_list = self.ground_truth_csv['Model ID'].to_list()

    def _preprocess_pc(self, filename):
        pointcloud = load_point_cloud(filename)
        if self.pc_transforms:
            pointcloud = self.pc_transforms(pointcloud)
        return pointcloud

    def _preprocess_text(self, input_text):
        encoded_dict = self.text_tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = torch.tensor(encoded_dict['input_ids'])
        attention_mask = torch.tensor(encoded_dict['attention_mask'])
        return input_ids, attention_mask

    def __len__(self):
        return len(self.text_queries_list)

    def __getitem__(self, idx):
        text_queries_id = self.text_queries_list[idx]
        point_cloud_id = self.point_cloud_list[idx]

        text_sample = self.text_queries_mapping[text_queries_id]
        text_input_ids, text_attention_mask = self._preprocess_text(text_sample)

        pc_path = os.path.join(self.point_cloud_path, f'{point_cloud_id}.obj')

        point_cloud = self._preprocess_pc(pc_path)

        return {"point_cloud": point_cloud, "input_ids": text_input_ids, "attention_mask": text_attention_mask}

    def collate_fn(self, batch):
        batch_as_dict = {
            "point_clouds": torch.stack([x["point_cloud"] for x in batch]),
            "sketch_images": torch.stack([x["sketch_image"] for x in batch]),
        }

        return batch_as_dict