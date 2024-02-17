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
                param.grad /= self.factor
                if torch.isinf(param.grad).sum() > 0 or torch.isnan(param.grad).sum() > 0:
                    logging.warning(f"Faced grad overflow with factor = {self.factor}")
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

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        if precision == "full":
            outputs = model(images)
            loss = criterion(outputs, labels)
        else:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        # code for loss scaling
        optimizer.zero_grad()
        if loss_scaling != "none":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        else:
            loss.backward()
            optimizer.step()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(
        batch_size: int = 32,
        precision: Literal["full", "half"] = "full",
        loss_scaling: Literal["none", "static", "dynamic"] = "none",
        scale_factor: int = 128):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data(batch_size)

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device, precision, loss_scaling, scale_factor)
