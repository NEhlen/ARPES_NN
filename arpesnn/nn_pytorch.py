import os
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

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
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_workers: int | None = None,
    use_amp: bool = True,
    loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        torch.nn.CrossEntropyLoss()
    ),
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and device.type == "cuda"
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    network = network.to(device)
    dataset = TensorDataset(in_data, target_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"Training on device: {device}")
    print(
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"num_workers={num_workers}, amp={amp_enabled}"
    )

    last_epoch_loss = 0.0
    steps_per_epoch = len(dataloader)
    log_every = max(1, steps_per_epoch // 10)

    for epoch in range(epochs):
        network.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        seen_samples = 0

        for step, (inputs, targets) in enumerate(dataloader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = network(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_n = inputs.size(0)
            seen_samples += batch_n
            epoch_loss += loss.item() * batch_n

            if step % log_every == 0 or step == steps_per_epoch:
                print(
                    f"epoch {epoch + 1}/{epochs} step {step}/{steps_per_epoch} "
                    f"batch_loss={loss.item():.6f}"
                )

        last_epoch_loss = epoch_loss / seen_samples
        elapsed = time.perf_counter() - epoch_start
        samples_per_sec = seen_samples / elapsed if elapsed > 0 else 0.0
        epoch_log = (
            f"epoch {epoch + 1}/{epochs} done "
            f"loss={last_epoch_loss:.6f} "
            f"time={elapsed:.2f}s "
            f"throughput={samples_per_sec:.2f} samples/s"
        )
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            epoch_log += f" peak_mem={peak_mem_gb:.2f}GB"
            torch.cuda.reset_peak_memory_stats(device)
        print(epoch_log)

    return last_epoch_loss


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
        arpes_unet,
        in_data,
        target_data,
        loss_fn=torch.nn.MSELoss(),
        epochs=100,
        batch_size=8,
        lr=1e-4,
    )
    torch.save(
        arpes_unet.state_dict(),
        os.path.join(models_path, "test_model"),
    )
    print(final_loss)
