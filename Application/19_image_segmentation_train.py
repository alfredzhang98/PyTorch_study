
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json

from tqdm import tqdm
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CatSegmentationDataset(Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(self, images_dir, images_size=32):
        
        images_dir = os.path.abspath(images_dir)
        print("Reading data from:", images_dir)
        image_root_path = os.path.join(images_dir, "JPEGImages")
        mask_root_path = os.path.join(images_dir, "SegmentationClassPNG")
        if not os.path.isdir(image_root_path):
            raise FileNotFoundError(f"JPEGImages not found: {image_root_path}")
        if not os.path.isdir(mask_root_path):
            raise FileNotFoundError(f"SegmentationClassPNG not found: {mask_root_path}")

        self.image_slices = []
        self.mask_slices = []

        for im_name in os.listdir(image_root_path):
            mask_name = im_name.split(".")[0] + ".png"

            # os.sep is the system path separator, e.g., '/' for Linux and '\' for Windows
            image_path = os.path.join(image_root_path, im_name)
            mask_path = os.path.join(mask_root_path, mask_name)

            im = np.asarray(Image.open(image_path).resize((images_size, images_size)))
            mask = np.asarray(Image.open(mask_path).resize((images_size, images_size)))

            self.image_slices.append(im / 255.0)
            self.mask_slices.append(mask)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        image =  image.transpose((2, 0, 1))  # HWC -> CHW
        mask = mask[np.newaxis, ...]  # HW -> CHW

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask


def data_loaders(args):
    dataset_train = CatSegmentationDataset(
        images_dir=args.images,
        images_size=args.image_size
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    return loader_train

class DiceLoss(nn.Module):
    def __init__(self, class_weights=None, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = float(smooth)

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w)
        else:
            self.register_buffer("class_weights", None)



    def forward(self, y_pred, y_true):
        # ones = torch.ones(1, device=y_pred.device, dtype=y_pred.dtype)
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

from unet import UNet

def makedirs(args):
    os.makedirs(args.ckpts, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def audit_devices(model):
    bad = []
    for name, p in model.named_parameters():
        if p.device.type == "cpu":
            bad.append(("param", name, tuple(p.shape), p.device))
    for name, b in model.named_buffers():
        if b.device.type == "cpu":
            bad.append(("buffer", name, tuple(b.shape), b.device))
    return bad


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train = data_loaders(args)

    unet = UNet(in_channels=CatSegmentationDataset.in_channels, out_channels=CatSegmentationDataset.out_channels)
    unet = unet.to(device)

    problems = audit_devices(unet)
    if problems:
        print("Still on CPU before forward():")
        for kind, name, shape, dev in problems:
            print(f" - {kind:6s} {name:40s} {shape} @ {dev}")
    else:
        print("All registered params/buffers moved to", device)

    dsc_loss = DiceLoss().to(device)

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    loss_train = []

    step = 0
    last_loss = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        unet.train()
        for batch_idx, (data, target) in enumerate(loader_train):
            step += 1
            data   = torch.as_tensor(data, device=device)
            target = torch.as_tensor(target, device=device)

            y_pred = unet(data)
            optimizer.zero_grad()
            loss = dsc_loss(y_pred, target)

            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

            if (step) % 10 == 0:
                print('Step', step, 'Training Loss:', np.mean(loss_train[-10:]))

        if last_loss == 0 or np.mean(loss_train) < last_loss:
            # remove the last saved model to save space
            if last_loss != 0:
                try:
                    os.remove(args.ckpts + '/unet_epoch_{}_{}.pth'.format(epoch-1, int(last_loss * 1000)))
                except Exception as e:
                    print("Error removing previous checkpoint:", e)
            last_loss = np.mean(loss_train)

            torch.save(unet, args.ckpts + '/unet_epoch_{}_{}.pth'.format(epoch, int(last_loss * 1000)))


# 在 Notebook 中使用 argparse 不方便，这里直接构造一个默认参数对象
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
args = argparse.Namespace(
    batch_size=16,      # Batch Size
    epochs=100,          # Epoch number
    lr=0.0001,          # Learning rate
    device="cuda:0",   # Device for training (auto-falls back to CPU if no CUDA)
    num_workers=4,      # Dataloader workers
    ckpts=os.path.join(_BASE_DIR, "ckpts", "work2"),   # folder to save weights
    logs=os.path.join(_BASE_DIR, "logs"),               # folder to save logs
    images=os.path.join(_BASE_DIR, "data", "work2"),   # should contain JPEGImages & SegmentationClassPNG
    image_size=256      # target input image size
)


# 运行训练
main(args)
