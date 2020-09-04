import random
from pathlib import Path


file_dir = Path("data/train")
image_id = set(
    map(
        lambda x: str(x.parts[-1]).replace("_mask", "").split(".")[0],
        file_dir.rglob("*.jpg"),
    )
)

pos_image_list, neg_image_list = [], []
for id in image_id:
    if file_dir.joinpath(id + "_mask.jpg").exists():
        pos_image_list.append(id)
    else:
        neg_image_list.append(id)

random.seed(1234)
random.shuffle(pos_image_list)
random.seed(1234)
random.shuffle(neg_image_list)

with open(file_dir.joinpath("label.csv"), "w") as f:
    f.write("id,label,mode\n")

    for i, pos_id in enumerate(pos_image_list):
        if i <= int(0.8 * len(pos_image_list)):
            f.write(pos_id + ",1,train\n")
        else:
            f.write(pos_id + ",1,val\n")
    for j, neg_id in enumerate(neg_image_list):
        if j <= int(0.8 * len(neg_image_list)):
            f.write(neg_id + ",0,train\n")
        else:
            f.write(neg_id + ",0,val\n")
