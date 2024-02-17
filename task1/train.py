import argparse
import logging
import torch
from torch import nn
from tqdm.auto import tqdm
from typing import Literal, Union

from unet import Unet
from dataset import get_train_data


class Scaler:
    def __init__(self, scale_factor: int):
        self.factor = scale_factor

    def scale(self, loss: torch.Tensor):
        return self.factor * loss

    def step(self, optimizer: torch.optim.Optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.grad = param.grad / self.factor
                if torch.isinf(param.grad).sum() > 0 or torch.isnan(param.grad).sum() > 0:
                    # logging.warning(f"Faced grad overflow with factor = {self.factor}")
                    return False
        optimizer.step()
        return True


class DynamicScaler(Scaler):
    def step(self, optimizer):
        if super().step(optimizer):
            self.factor *= 2
        else:
            self.factor /= 2


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: Literal["full", "half"] = "full",
    loss_scaling: Literal["none", "static", "dynamic"] = "none",
    scale_factor: int = 128
) -> None:
    model.train()

    if loss_scaling == "static":
        scaler = Scaler(scale_factor=scale_factor)
    elif loss_scaling == "dynamic":
        scaler = DynamicScaler(scale_factor=scale_factor)

    precis = torch.float16 if precision == "half" else torch.float32
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(dtype=precis):
            outputs = model(images)
            loss = criterion(outputs, labels)

            # code for loss scaling
            optimizer.zero_grad()
            if loss_scaling != "none":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if loss_scaling != "none":
            scaler.step(optimizer)
        else:
            optimizer.step()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(
        batch_size: int = 32,
        num_epochs: int = 5,
        precision: Literal["full", "half"] = "full",
        loss_scaling: Literal["none", "static", "dynamic"] = "none",
        scale_factor: int = 128):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data(batch_size)

    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device, precision, loss_scaling, scale_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--precision", type=str, default="full")
    parser.add_argument("--loss_scaling", type=str, default="none")
    parser.add_argument("--scale_factor", type=float, default=128)

    args = parser.parse_args()

    train(args.batch_size, args.num_epochs, args.precision, args.loss_scaling, args.scale_factor)
