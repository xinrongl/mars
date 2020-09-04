import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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

model = Res_Unet(n_channels=3, n_classes=1)

os.system("git add .")
os.system("git commit -m 'finish training'")
os.system("git push origin master")
