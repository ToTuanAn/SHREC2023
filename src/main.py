import hydra
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.dataset.text_pc_dataset import TextPointCloudDataset
from src.extractor.bert_extractor import LangExtractor
from src.model import MODEL_REGISTRY
from src.model.text_pc_baseline import TextPointCloudNetwork

from src.transformer.pc_transformer import Normalize, RandRotation_z, RandomNoise, ToTensor

train_transforms = transforms.Compose([
                        Normalize(),
                        RandRotation_z(),
                        RandomNoise(),
                        ToTensor()
                    ])

EPOCHS = 100

#@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main():
    train_set = TextPointCloudDataset(root_dir='/home/totuanan/Workplace/SHREC2023/SHREC2023/dataset', pc_transform=train_transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=False)

    model = TextPointCloudNetwork(num_classes=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)


    for epoch in tqdm(range(EPOCHS)):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            point_cloud, input_ids, attention_mask = data['point_cloud'].float().transpose(1,2), data['input_ids'].squeeze(0), data['attention_mask'].squeeze(0)
            point_cloud_embedding, text_queries_embedding = model.forward({'point_cloud': point_cloud,
                                                                          'text_queries': {
                                                                            'input_ids': input_ids,
                                                                            'attention_mask': attention_mask
                                                                          }
                                                                         })
            print(point_cloud_embedding.shape, text_queries_embedding.shape)
            break
        break


if __name__ == '__main__':
    main()

