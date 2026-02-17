import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()


# unet with skip connections
class Unet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # general layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # decoder
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.skipconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.skipconv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.out = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # encoder
        # block 1
        xe1 = self.conv1(x)  # outputs 256x256x128
        xe1 = self.bn1(xe1)
        xe1 = self.relu(xe1)

        # block 2
        xe2 = self.conv2(xe1)  # outputs 256x256x64
        xe2 = self.bn2(xe2)
        xe2 = self.relu(xe2)

        # maxpool
        xp = self.pool(xe2)  # outputs 128x128x64

        # block 3
        xe3 = self.conv3(xp)  # outputs 128x128x32
        xe3 = self.bn3(xe3)
        xe3 = self.relu(xe3)

        # bottleneck
        xb = self.pool(xe3)  # outputs 64x64x32

        # decoder
        # block 1
        xd1 = self.upconv1(xb)  # outputs 128x128x32
        xd1 = torch.cat([xd1, xe3], dim=1)  # outputs 128x128x64
        xd1 = self.skipconv1(xd1)  # outputs 128x128x64
        xd1 = self.relu(xd1)
        # block 2
        xd2 = self.upconv2(xd1)  # outputs 256x256x64
        xd2 = torch.cat([xd2, xe2], dim=1)  # outptts 256x256x128
        xd2 = self.skipconv2(xd2)  # outputs 256x256x128
        xd2 = self.relu(xd2)

        return self.out(xd2)  # outputs 256x256x1


def load_data(base_folder: str):
    input_list = []
    output_list = []
    for fldr in os.listdir(base_folder):
        if fldr != ".gitkeep":
            for file in os.listdir(base_folder + "/" + fldr):
                if file.split("_")[-1] == "input.txt":
                    input_list.append(np.loadtxt(base_folder + "/" + fldr + "/" + file))
                elif file.split("_")[-1] == "target.txt":
                    output_list.append(
                        np.loadtxt(base_folder + "/" + fldr + "/" + file)
                    )
    return np.array(input_list), np.array(output_list)


def train(
    network: nn.Module,
    in_data: torch.Tensor,
    target_data: torch.Tensor,
    epochs=2,
    loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        torch.nn.CrossEntropyLoss()
    ),
):
    dataset = TensorDataset(in_data, target_data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)

    for e in range(epochs):
        print("EPOCH:", e)
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(dataloader):
            inputs, targets = data

            optimizer.zero_grad()

            outputs = network(inputs)

            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 5 == 0:
                if i != 0:
                    last_loss = running_loss / 5  # loss per batch
                else:
                    last_loss = running_loss
                print("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0

    return last_loss


if __name__ == "__main__":
    dataset_path = os.getenv("DATASET_PATH")
    models_path = os.getenv("MODELS_PATH")
    if not dataset_path:
        raise ValueError("DATASET_PATH is not set. Add it to your .env file.")
    if not models_path:
        raise ValueError("MODELS_PATH is not set. Add it to your .env file.")

    in_data, target_data = load_data(dataset_path)
    in_data = torch.tensor(in_data, dtype=torch.float32).unsqueeze(1)
    target_data = torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)
    arpes_unet = Unet()
    final_loss = train(
        arpes_unet, in_data, target_data, loss_fn=torch.nn.MSELoss(), epochs=100
    )
    torch.save(
        arpes_unet.state_dict(),
        os.path.join(models_path, "test_model"),
    )
    print(final_loss)
