# split data according to segmentation-models-pytorch
import random
from pathlib import Path
import shutil
import sys


file_dir = Path(sys.argv[1])  # data/train
image_id = list(
    set(
        map(
            lambda x: str(x.parts[-1]).replace("_mask", "").split(".")[0],
            file_dir.rglob("*.jpg"),
        )
    )
)

x_train_dir = file_dir.joinpath("train")
y_train_dir = file_dir.joinpath("trainannot")
x_val_dir = file_dir.joinpath("val")
y_val_dir = file_dir.joinpath("valannot")
dirs = [x_train_dir, x_val_dir, y_train_dir, y_val_dir]
for d in dirs:
    d.mkdir(exist_ok=True, parents=True)

random.seed(1234)
random.shuffle(image_id)
split_index = int(0.8 * len(image_id))
for i, image_id in enumerate(image_id):
    try:
        image_fname = f"{image_id}.jpg"
        mask_fname = f"{image_id}_mask.jpg"

        if i <= split_index:
            shutil.move(
                file_dir.joinpath(image_fname), x_train_dir.joinpath(image_fname)
            )
            shutil.move(file_dir.joinpath(mask_fname), y_train_dir.joinpath(mask_fname))
        else:
            shutil.move(file_dir.joinpath(image_fname), x_val_dir.joinpath(image_fname))
            shutil.move(file_dir.joinpath(mask_fname), y_val_dir.joinpath(mask_fname))
    except FileNotFoundError:
        print(f"image id {image_id} not found")
