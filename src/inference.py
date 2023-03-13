import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset.inference_dataset import InferenceDataset
from src.utils.opt import Opts
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import MODEL_REGISTRY

from pathlib import Path

from src.utils.pc_transform import Normalize, ToTensor


def inference(cfg, pretrained_ckpt=None):
    test_transforms = transforms.Compose([ Normalize(),
                                            ToTensor()])

    test_set = InferenceDataset(root_dir='../dataset', pc_transform=test_transforms, stage='train')
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    model = MODEL_REGISTRY.get(cfg["model"]["name"])(cfg)
    model = model.load_from_checkpoint(pretrained_ckpt, cfg=cfg, strict=True)
    model.eval()

    inference_result = dict()

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader, 0)):
            point_cloud_id, point_cloud, text_id, text_input_ids, \
                text_attention_mask = data["point_cloud_id"],  \
                                        data["point_cloud"].float(), str(data["text_id"]), data["input_ids"], data["attention_mask"]
            output = model({"true_point_clouds": point_cloud.transpose(1,2),
                                     "false_point_clouds": point_cloud.transpose(1,2),
                                     "text_queries": {
                                         "input_ids": text_input_ids,
                                         "attention_mask": text_attention_mask
                                     }})

            distance = F.pairwise_distance(output['true_pc_embedding_feats'], output['text_query_embedding_feats'])

            if text_id not in inference_result:
                inference_result[text_id] = []

            inference_result[text_id].append((point_cloud_id, distance))

    df = pd.DataFrame(inference_result.items())
    df.to_csv('result.csv')


if __name__ == '__main__':
    pretrained_ckpt = "/home/totuanan/Workplace/SHREC2023/SHREC2023/runs/hcmus-shrec23-textANIMAR/gj7yo8qj/checkpoints/baseline-epoch=1270-train_loss=0.2654-val_loss=0.0820.ckpt"
    cfg_path = '/home/totuanan/Workplace/SHREC2023/SHREC2023/configs/template.yml'

    cfg = Opts(cfg=cfg_path).parse_args()
    inference(cfg, pretrained_ckpt=pretrained_ckpt)