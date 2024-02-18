import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import pandas as pd

from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Settings, Clothes, seed_everything
from vit import ViT


def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders(shuffle_train: bool) -> tp.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    train_frame = frame.sample(frac=Settings.train_frac)
    val_frame = frame.drop(train_frame.index)

    train_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", train_frame, transform=train_transforms
    )
    val_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", val_frame, transform=val_transforms
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=Settings.batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(dataset=val_data, batch_size=Settings.batch_size, shuffle=False)

    return train_loader, val_loader


def make_step(data, label, model, criterion, optimizer, is_train=True):
    data = data.to(Settings.device)
    label = label.to(Settings.device)
    output = model(data)
    loss = criterion(output, label)
    acc = (output.argmax(dim=1) == label).float().mean()
    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss, acc


def run_epoch(model, train_loader, val_loader, criterion, optimizer, prof=None) -> tp.Tuple[float, float]:
    n_steps = len(train_loader) if prof is None else Settings.n_steps
    epoch_loss, epoch_accuracy = 0, 0
    val_loss, val_accuracy = 0, 0
    model.train()
    for step, (data, label) in enumerate(tqdm(train_loader, desc="Train")):
        if prof is not None and step == Settings.n_steps:
            break
        loss, acc = make_step(data, label, model, criterion, optimizer, is_train=True)
        epoch_accuracy += acc.item() / len(train_loader)
        epoch_loss += loss.item() / len(train_loader)
        if prof is not None:
            prof.step()

    if prof is not None:
        return epoch_loss, epoch_accuracy

    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        loss, acc = make_step(data, label, model, criterion, optimizer, is_train=False)
        val_accuracy += acc.item() / len(train_loader)
        val_loss += loss.item() / len(train_loader)

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy


def main():
    seed_everything()
    model = get_vit_model()
    train_loader, val_loader = get_loaders(shuffle_train=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)

    # data, label = next(iter(train_loader))
    # with profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/forward"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof_1:
    #     with record_function("model_forward"):
    #         data = data.to(Settings.device)
    #         label = label.to(Settings.device)
    #         output = model(data)
    #         loss = criterion(output, label)
    #
    # with profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/backward"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof_2:
    #     with record_function("model_backward"):
    #         loss.backward()

    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/several_iter"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof_3:
        with record_function("several_iter"):
            run_epoch(model, train_loader, val_loader, criterion, optimizer, prof_3)


if __name__ == "__main__":
    main()
