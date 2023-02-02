import os
import numpy as np
import itertools
import math, random

random.seed = 42

import torch
import fire
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from loader.pointnetloader import PointSampler, Normalize, ToTensor


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])


def preprocessing_pc(off_path='./dataset/PC_OFF/'):
    for file in os.listdir(off_path):
        with open(os.path.join(off_path, file), 'r') as f:
            verts, faces = read_off(f)

        pointcloud = PointSampler(3000)((verts, faces))
        pointcloud = Normalize()(pointcloud)
        with open(f'./dataset/PC/{file[:-4]}.txt', 'w+') as f:
            for triple_points in pointcloud:
                f.write(str(triple_points[0]) + ' ' + str(triple_points[1]) + ' ' + str(triple_points[2]) + '\n')



def preprocessing_img(img_path='./dataset/IMG_Sketch/'):
    pass


if __name__ == "__main__":
    fire.Fire(preprocessing_pc)
    fire.Fire(preprocessing_img)
