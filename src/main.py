import hydra
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.dataset.text_pc_dataset import TextPointCloudDataset
from src.extractor.bert_extractor import LangExtractor
from src.model import MODEL_REGISTRY
from src.model.text_pc_baseline import TextPointCloudNetwork

from src.transformer.pc_transformer import (
    Normalize,
    RandRotation_z,
    RandomNoise,
    ToTensor,
)

train_transforms = transforms.Compose(
    [Normalize(), RandRotation_z(), RandomNoise(), ToTensor()]
)

validation_transforms = transforms.Compose([Normalize(), ToTensor()])

EPOCHS = 100


# @hydra.main(version_base=None, config_path="../configs", config_name="config")
def main():
    train_set = TextPointCloudDataset(
        root_dir="/home/totuanan/Workplace/SHREC2023/SHREC2023/dataset",
        pc_transform=train_transforms,
        type="train",
    )
    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)

    validation_set = TextPointCloudDataset(
        root_dir="/home/totuanan/Workplace/SHREC2023/SHREC2023/dataset",
        pc_transform=validation_transforms,
        type="validation",
    )
    validation_loader = DataLoader(dataset=validation_set, batch_size=2, shuffle=True)

    model = TextPointCloudNetwork(num_classes=128)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

    for epoch in tqdm(range(EPOCHS)):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            true_point_cloud, false_point_cloud, input_ids, attention_mask = (
                data["true_point_cloud"].float().transpose(1, 2),
                data["false_point_cloud"].float().transpose(1, 2),
                data["input_ids"],
                data["attention_mask"],
            )
            optimizer.zero_grad()
            (
                true_point_cloud_embedding,
                false_point_cloud_embedding,
                text_queries_embedding,
            ) = model.forward(
                {
                    "true_point_cloud": true_point_cloud,
                    "false_point_cloud": false_point_cloud,
                    "text_queries": {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                }
            )

            loss = triplet_loss(
                text_queries_embedding,
                true_point_cloud_embedding,
                false_point_cloud_embedding,
            )
            loss.backward()
            optimizer.step()

            # print("True point cloud embedding: ", true_point_cloud_embedding)
            # print("False point cloud embedding: ", false_point_cloud_embedding)
            # print("Text queries embedding: ", text_queries_embedding)

            running_loss += loss.item()
            if i % 5 == 4:  # print every 10 mini-batches
                print(
                    "[Epoch: %d, Batch: %4d / %4d], loss: %.3f"
                    % (epoch + 1, i + 1, len(train_loader), running_loss / 10)
                )
                running_loss = 0.0

        for i, data in enumerate(validation_loader):
            true_point_cloud, false_point_cloud, input_ids, attention_mask = (
                data["true_point_cloud"].float().transpose(1, 2),
                data["false_point_cloud"].float().transpose(1, 2),
                data["input_ids"],
                data["attention_mask"],
            )

            (
                true_point_cloud_embedding,
                false_point_cloud_embedding,
                text_queries_embedding,
            ) = model.forward(
                {
                    "true_point_cloud": true_point_cloud,
                    "false_point_cloud": false_point_cloud,
                    "text_queries": {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                }
            )

            positive_eucledian_distance = F.pairwise_distance(
                true_point_cloud_embedding, text_queries_embedding
            )
            negative_eucledian_distance = F.pairwise_distance(
                false_point_cloud_embedding, text_queries_embedding
            )

            print("Positive Eucledian Distance:-", positive_eucledian_distance)
            print("Negative Eucledian Distance:-", negative_eucledian_distance)

        # save the model
        if (epoch) % 20 == 0:
            torch.save(
                model.state_dict(),
                f"/home/totuanan/Workplace/SHREC2023/SHREC2023/weights/save_{epoch}.pth",
            )


if __name__ == "__main__":
    main()
