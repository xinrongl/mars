import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from collections import OrderedDict

from unet.unet import Res_Unet
from utils.dataset import ImageDataset
from utils.logger import Logger

parser = argparse.ArgumentParser("Train UNnet")
parser.add_argument("-b", "--batch_size", default=128, type=int)
parser.add_argument("-r", "--resize", default=512, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr_step", default=5, type=int)
parser.add_argument("--lr_factor", default=0.1, type=int)
parser.add_argument("--num_epochs", default=15, type=int)
parser.add_argument("--log_freq", default=0, type=int)
parser.add_argument(
    "-wd", "--weight_decay", default=0.0, type=float, help="weight decay factor"
)
args = parser.parse_args()


timestamp = datetime.now()
model_path = Path(f"./models/{timestamp: %Y%m%d%H%M}")
model_path.mkdir(exist_ok=True, parents=True)


logger = Logger(level="DEBUG")
logger.set_stream_handler(level="INFO")
logger.set_file_handler(model_path.joinpath(f"{timestamp: %Y%m%d%H%M}.log"))
logger.debug(open(Path(__file__), "r").read())


torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def train_loop(
    epochs,
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    model_path,
):
    best_acc1 = 0
    train_acc_history = []
    val_acc_history = []
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch} LR: {lr_scheduler.get_last_lr()}")
        train_score = train(
            train_loader, model, criterion, optimizer, epoch, model_path, logging=True
        )

        lr_scheduler.step()

        train_acc_history.append(train_score)
        val_acc = validate(val_loader, model, criterion)
        val_acc_history.append(val_acc)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)
        if is_best and epoch >= 15:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": "inception",
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filename=f"{model_path}/checkpoint.pth.tar",
            )
    return np.asarray(train_acc_history), np.asarray(val_acc_history)


def train(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    avg_f1 = AverageMeter("F1", ":6.2f")
    avg_acc = AverageMeter("Acc", ":6.2f")

    model.train()
    num_steps = len(train_loader)

    end = time.time()

    for i, (input_, targets) in enumerate(train_loader):
        if i >= num_steps:
            break
        output = model(input_.to(device))
        loss = criterion(output, targets.to(device))

        _, predicts = torch.max(output.data, 1)
        predicts = predicts.cpu().numpy()
        targets = targets.cpu().numpy()
        avg_f1.update(f1_score(targets, predicts, average="micro"))
        avg_acc.update(accuracy_score(targets, predicts))

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(f"*\tAccuracy[train]: {avg_acc.avg:.4f}\tLoss[train]: {losses.avg:.4f}")
    return avg_acc.avg


def validate(val_loader, model, criterion):
    """ Returns predictions and targets, if any. """

    avg_losses = AverageMeter("Loss", ":.4e")
    avg_acc = AverageMeter("Acc", ":6.2f")

    model.eval()

    all_predicts, all_confs, all_targets = [], [], []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input_, target = data

            output = model(input_.to(device))
            loss = criterion(output.to(device), target.to(device))
            avg_losses.update(loss)

            all_confs.append(output)
            _, predicts = torch.max(output.data, 1)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    predicts = predicts.cpu().numpy()
    confs = confs.cpu().numpy()
    targets = targets.cpu().numpy()

    acc = accuracy_score(targets, predicts)
    avg_acc.update(acc)
    logger.info(
        f"*\tAccuracy[valid]: {avg_acc.avg:.4f}\tLoss[valid]: {avg_losses.avg:.4f}"
    )

    return acc


def load_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        tmp = key
        state_dict[tmp] = value
    model.load_state_dict(state_dict)
    return model


train_dataset = ImageDataset(
    image_dir="data/train", label_fname="data/train/label.csv", mode="train"
)
val_dataset = ImageDataset(
    image_dir="data/train", label_fname="data/train/label.csv", mode="val"
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)


model = Res_Unet(n_channels=3, n_classes=1)
model = nn.DataParallel(model)
model.to(device)

criterion = nn.BCELoss()
optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.2, patience=5, verbose=False
)
train_res, val_res = train_loop(
    args.epochs, train_loader, val_loader, model, criterion, optimizer, lr_scheduler
)

os.system("git add .")
os.system("git commit -m 'finish training'")
os.system("git push origin master")
