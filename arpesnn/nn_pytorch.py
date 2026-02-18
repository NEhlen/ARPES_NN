import os
import copy
import csv
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset, random_split

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
                path = base_folder + "/" + fldr + "/" + file
                suffix = file.split("_")[-1]
                if suffix == "input.txt":
                    input_list.append(np.loadtxt(path))
                elif suffix == "target.txt":
                    output_list.append(np.loadtxt(path))
                elif suffix == "input.npy":
                    input_list.append(np.load(path))
                elif suffix == "target.npy":
                    output_list.append(np.load(path))
    return np.array(input_list), np.array(output_list)


def evaluate(
    network: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    amp_enabled: bool,
) -> float:
    network.eval()
    total_loss = 0.0
    seen_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = network(inputs)
                data_loss = loss_fn(outputs, targets)
                reg_loss = (
                    l1_lambda * l1_regularization(network)
                    if l1_lambda > 0.0
                    else torch.zeros((), device=device)
                )
                loss = data_loss + reg_loss
            batch_n = inputs.size(0)
            seen_samples += batch_n
            total_loss += loss.item() * batch_n
    return total_loss / max(seen_samples, 1)


def l1_regularization(model: nn.Module) -> torch.Tensor:
    l1_penalty = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        l1_penalty = l1_penalty + param.abs().sum()
    return l1_penalty


def train(
    network: nn.Module,
    in_data: torch.Tensor,
    target_data: torch.Tensor,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-4,
    val_split: float = 0.2,
    early_stopping_patience: int = 15,
    weight_decay: float = 1e-4,
    l1_lambda: float = 0.0,
    num_workers: int | None = None,
    use_amp: bool = True,
    random_seed: int = 42,
    model_save_path: str | None = None,
    metrics_csv_path: str | None = None,
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
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples for training after applying val_split.")
    generator = torch.Generator().manual_seed(random_seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = dataset
        val_dataset = TensorDataset(
            torch.empty((0, *in_data.shape[1:]), dtype=in_data.dtype),
            torch.empty((0, *target_data.shape[1:]), dtype=target_data.dtype),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=lr, weight_decay=weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"Training on device: {device}")
    print(
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"weight_decay={weight_decay}, l1_lambda={l1_lambda}, "
        f"num_workers={num_workers}, amp={amp_enabled}, "
        f"train_samples={train_size}, val_samples={val_size}"
    )
    if metrics_csv_path is not None:
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_data_loss",
                    "train_reg_loss",
                    "val_loss",
                    "time_sec",
                    "throughput_samples_per_sec",
                    "best_val_loss",
                    "is_best",
                ]
            )
        print(f"logging training metrics to {metrics_csv_path}")

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_state_dict = copy.deepcopy(network.state_dict())
    epochs_without_improvement = 0
    steps_per_epoch = len(train_loader)
    log_every = max(1, steps_per_epoch // 10)

    for epoch in range(epochs):
        network.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_reg_loss = 0.0
        seen_samples = 0

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = network(inputs)
                data_loss = loss_fn(outputs, targets)
                reg_loss = (
                    l1_lambda * l1_regularization(network)
                    if l1_lambda > 0.0
                    else torch.zeros((), device=device)
                )
                loss = data_loss + reg_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_n = inputs.size(0)
            seen_samples += batch_n
            epoch_loss += loss.item() * batch_n
            epoch_data_loss += data_loss.item() * batch_n
            epoch_reg_loss += reg_loss.item() * batch_n

            if step % log_every == 0 or step == steps_per_epoch:
                print(
                    f"epoch {epoch + 1}/{epochs} step {step}/{steps_per_epoch} "
                    f"batch_loss={loss.item():.6f} data_loss={data_loss.item():.6f}"
                )

        train_loss = epoch_loss / max(seen_samples, 1)
        train_data_loss = epoch_data_loss / max(seen_samples, 1)
        train_reg_loss = epoch_reg_loss / max(seen_samples, 1)
        val_loss = (
            evaluate(network, val_loader, device, loss_fn, amp_enabled)
            if val_size > 0
            else train_loss
        )
        elapsed = time.perf_counter() - epoch_start
        samples_per_sec = seen_samples / elapsed if elapsed > 0 else 0.0
        epoch_log = (
            f"epoch {epoch + 1}/{epochs} done "
            f"train_loss={train_loss:.6f} "
            f"data_loss={train_data_loss:.6f} "
            f"reg_loss={train_reg_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"time={elapsed:.2f}s "
            f"throughput={samples_per_sec:.2f} samples/s"
        )
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            epoch_log += f" peak_mem={peak_mem_gb:.2f}GB"
            torch.cuda.reset_peak_memory_stats(device)
        print(epoch_log)

        is_best = 0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_state_dict = copy.deepcopy(network.state_dict())
            epochs_without_improvement = 0
            is_best = 1
            if model_save_path is not None:
                torch.save(best_state_dict, model_save_path)
                print(
                    f"saved new best checkpoint to {model_save_path} "
                    f"(val_loss={best_val_loss:.6f})"
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"early stopping at epoch {epoch + 1}; "
                    f"no val improvement for {early_stopping_patience} epochs"
                )
                if metrics_csv_path is not None:
                    with open(metrics_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                epoch + 1,
                                f"{train_loss:.8f}",
                                f"{train_data_loss:.8f}",
                                f"{train_reg_loss:.8f}",
                                f"{val_loss:.8f}",
                                f"{elapsed:.4f}",
                                f"{samples_per_sec:.4f}",
                                f"{best_val_loss:.8f}",
                                is_best,
                            ]
                        )
                break
        if metrics_csv_path is not None:
            with open(metrics_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        f"{train_loss:.8f}",
                        f"{train_data_loss:.8f}",
                        f"{train_reg_loss:.8f}",
                        f"{val_loss:.8f}",
                        f"{elapsed:.4f}",
                        f"{samples_per_sec:.4f}",
                        f"{best_val_loss:.8f}",
                        is_best,
                    ]
                )

    network.load_state_dict(best_state_dict)
    return best_train_loss


if __name__ == "__main__":
    dataset_path = os.getenv("DATASET_PATH")
    models_path = os.getenv("MODELS_PATH")
    if not dataset_path:
        raise ValueError("DATASET_PATH is not set. Add it to your .env file.")
    if not models_path:
        raise ValueError("MODELS_PATH is not set. Add it to your .env file.")
    weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    l1_lambda = float(os.getenv("L1_LAMBDA", "0.0"))

    in_data, target_data = load_data(dataset_path)
    in_data = torch.tensor(in_data, dtype=torch.float32).unsqueeze(1)
    target_data = torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)
    arpes_unet = Unet()
    final_loss = train(
        arpes_unet,
        in_data,
        target_data,
        loss_fn=torch.nn.MSELoss(),
        epochs=100000,
        batch_size=8,
        lr=1e-4,
        val_split=0.3,
        early_stopping_patience=100000,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        model_save_path=os.path.join(models_path, "best_model.pt"),
        metrics_csv_path=os.path.join(models_path, "training_metrics.csv"),
    )
    print(f"best train loss: {final_loss:.6f}")
