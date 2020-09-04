from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir, label_fname, mode, resize=512):
        label_df = pd.read_csv(label_fname)
        self.mode = mode
        assert self.mode in ["train", "val"]
        self.resize = resize
        self.label_df = label_df[
            (label_df["mode"] == self.mode) & (label_df["label"] == 1)
        ]
        self.label_df.reset_index(drop=True, inplace=True)
        self.image_dir = Path(image_dir)

        transforms_list = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.resize),
        ]
        transforms_list.extend([transforms.ToTensor()])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index: int):
        data = self.label_df.iloc[index]
        image = Image.open(self.image_dir.joinpath(f"{data.id}.jpg"))
        mask_np = np.array(Image.open(self.image_dir.joinpath(f"{data.id}_mask.jpg")))
        mask_np = np.where(
            mask_np <= 9, np.uint8(0), np.uint8(255)
        )  # deal with changes in pixel on the edge in mannual labeling process
        # mask = Image.fromarray(mask_np)

        image = self.transforms(image)
        mask = torch.as_tensor(mask_np, dtype=torch.uint8)
        return image, mask

    def __len__(self) -> int:
        return len(self.label_df)
