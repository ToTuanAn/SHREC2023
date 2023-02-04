import os
import numpy as np
import itertools
import math, random
random.seed = 42

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from pointnetloader import ToTensor


class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=None):
        self.root_dir = root_dir

        self.transforms = transform if not valid else None
        self.valid = valid
        self.point_cloud_files = []
        self.image_sketch_files = []

        point_cloud_path = os.path.join(root_dir, 'PC')
        img_sketch_path = os.path.join(root_dir, 'IMG_Sketch')

        for files in sorted(os.listdir(point_cloud_path)):
            self.point_cloud_files.append(os.path.join(point_cloud_path, files))

        for files in sorted(os.listdir(img_sketch_path)):
            self.image_sketch_files.append(os.path.join(img_sketch_path, files))

    def __preproc__(self, file, is_pc):
        if is_pc:
            pass
        else:
            pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc_path = self.point_cloud_files[idx]
        img_path = self.image_sketch_files[idx]
        with open(pc_path, 'r') as f:
            point_cloud = self.__preproc__(f, True)

        with open(img_path, 'r') as f:
            sketch_image = self.__preproc__(f, False)

        return {'point_cloud': point_cloud,
                'sketch_image': sketch_image}