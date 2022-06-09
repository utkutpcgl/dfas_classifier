from pathlib import Path
import torch
from torchvision import datasets, transforms
import yaml

# input image size settings
IMG_RES_DICT = {"B0": 224, "B3": 300}
# TODO change this path
DATA_DIR = Path(f"/raid/utku/seq_dataset_baris")

with open("base_resnet18_hyperparams.yaml", "r") as reader:
    hyps = yaml.safe_load(reader)

# IMG_RES = IMG_RES_DICT[hyps["model_type"]]

data_transforms = {
    "train": transforms.Compose(
        [
            # transforms.Resize(IMG_RES_B0),
            # transforms.Resize((IMG_RES,IMG_RES)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(), #NOTE this worsens the results most probably.
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ),
    "val": transforms.Compose(
        [
            # Maybe apply random resized crop.
            # transforms.Resize((IMG_RES,IMG_RES)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    ),
}

image_datasets = {
    d_type: datasets.ImageFolder(DATA_DIR / d_type, data_transforms[d_type]) for d_type in ["train", "val"]
}

# pin_memory=True to speed up host to device data transfer with page-locked memory
dataloaders = {
    d_type: torch.utils.data.DataLoader(
        image_datasets[d_type], batch_size=hyps["batch_size"], shuffle=True, num_workers=8, pin_memory=True
    )
    for d_type in ["train", "val"]
}

dataset_sizes = {d_type: len(image_datasets[d_type]) for d_type in ["train", "val"]}
class_names = image_datasets["train"].classes
