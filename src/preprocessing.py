import os
import random

import numpy as np

from src.transformer.pc_transformer import PointSampler, Normalize
import pymeshlab
import numpy as np
import random

random.seed = 42

import fire


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def preprocessing_pc(off_path='/home/totuanan/Workplace/SHREC2023/SHREC2023/dataset/PC_OFF/'):
    for file in os.listdir(off_path):
        with open(os.path.join(off_path, file), 'r') as f:
            verts, faces = read_off(f)

        pointcloud = PointSampler(3000)((verts, faces))
        with open(f'/home/totuanan/Workplace/SHREC2023/SHREC2023/dataset/pointcloud/{file[:-4]}.txt', 'w+') as f:
            for triple_points in pointcloud:
                f.write(str(triple_points[0]) + ' ' + str(triple_points[1]) + ' ' + str(triple_points[2]) + '\n')



def load_point_cloud(filename: str, n_sample: int=1024):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    ms.generate_sampling_poisson_disk(samplenum=n_sample)
    pc_list = ms.current_mesh().vertex_matrix()

    n = len(pc_list)
    if n > n_sample:
        pc_list = np.array(random.choices(pc_list, k=n_sample))
    elif n < n_sample:
        pc_list = np.concatenate([pc_list, np.array(random.choices(pc_list, k=n_sample - n))], axis=0)

    return pc_list


def preprocessing_img(img_path='./dataset/IMG_Sketch/'):
    pass


if __name__ == "__main__":
    pc = load_point_cloud('/home/totuanan/Workplace/SHREC2023/SHREC2023/dataset/PC_OBJ/0a77cd2a91dfff53.obj')
    print(pc)

