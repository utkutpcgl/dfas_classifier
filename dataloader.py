from pathlib import Path
import torch
from torchvision import datasets, transforms
import yaml
import albumentations as album

# input image size settings
with open("hyperparameters.yaml", "r") as reader:
    HYPS = yaml.safe_load(reader)

IMG_RES_DICT = {"B0": 224, "B1": 240, "B3": 300}
IMG_RES_RESNET = 224
TEST_AVAILABLE = HYPS["TEST"]
IGNORE_OTHER = HYPS["IGNORE_OTHER"]
# TODO discarded "atış", "araçlı_atış", "siper_mevzi" for now
# C_DICT = {'araç': 0, 'insan': 1, 'askeri araç': 2, 'sivil araç': 3, 'askeri insan': 4, 'sivil insan': 5,
#           'lastikli': 6, 'paletli': 7, 'silahlı asker': 8, 'silahsız asker': 9, 'Top-lastikli': 10, 'lastikli araç': 11,
#           'Tank': 12, 'Top-paletli': 13, 'ZPT': 14, 'Tank-M48': 15, 'Tank-M60': 16, 'Tank-fırtına': 17, 'Tank-leopard': 18}

GPU_IDS = HYPS["GPUS"]
GPU_ID0 = GPU_IDS[0]
DEVICE = f"cuda:{GPU_ID0}"
ADVANCED_AUG = HYPS["ADV_AUG"]
TASK = HYPS["TASK"]  # either arac or atis.
DO_H_FLIP = HYPS["DO_H_FLIP"]  # either arac or atis.
DO_NORMALIZE = HYPS["DO_NORMALIZE"]
# Path is either "/home/kuartis-dgx1/utku/dfas_classifier/data_ops/{TASK}_classification_dataset_combined"
# OR "/home/utku/Documents/repos/dfas_classifier/data_ops/{TASK}_classification_dataset_combined"
DATA_DIR = Path(f"/home/utku/Documents/repos/dfas_classifier/data_ops/{TASK}_classification_dataset_combined")
DETECT_PATH = DATA_DIR / "detect"
if "effnet" in HYPS["MODEL"]:
    IMG_RES = IMG_RES_DICT[HYPS["model_type"]]
elif "resnet" in HYPS["MODEL"]:
    IMG_RES = IMG_RES_RESNET

# Transforms
# NOTE resize can work on PIL images and returns pil image.
RESIZE = transforms.Resize((IMG_RES, IMG_RES))
H_FLIP = transforms.RandomHorizontalFlip(p=0.3)
# Advanced transformations.
COLOR_JITTER = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.5)
RANDOM_PERSP = transforms.RandomPerspective(distortion_scale=0.3, p=0.2)
RANDOM_ROT = transforms.RandomRotation(degrees=10)
GAUS_BLUR = transforms.GaussianBlur(kernel_size=5)
RAND_POSTER = transforms.RandomPosterize(bits=6, p=0.2)
RAND_SHARP = transforms.RandomAdjustSharpness(sharpness_factor=0.7, p=0.2)
RAND_AUTOCONTR = transforms.RandomAutocontrast(p=0.2)
RAND_EQUALIZE = transforms.RandomEqualize(p=0.2)
# Final transformations.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NORMALIZE = transforms.Normalize(MEAN, STD)
TO_TENSOR = transforms.ToTensor()
RAND_ERASE = transforms.RandomErasing(
    p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False
)  # Apply after to tensor, unlike other transforms.


# NOTE RandomResizedCrop was not useful, but can work (do not zoom)
train_transforms = [RESIZE]
if DO_H_FLIP:
    train_transforms.append(H_FLIP)
if ADVANCED_AUG:
    train_transforms.extend(
        [
            # Advanced transformations.
            COLOR_JITTER,
            RANDOM_PERSP,
            RANDOM_ROT,
            GAUS_BLUR,
            RAND_POSTER,
            RAND_SHARP,
            RAND_AUTOCONTR,
            RAND_EQUALIZE,
        ]
    )
train_transforms.extend([TO_TENSOR, NORMALIZE])
if DO_NORMALIZE:
    # NOTE normalization is curcial (created 20% accuracy difference.)
    train_transforms.append(NORMALIZE)
if ADVANCED_AUG:
    train_transforms.append(RAND_ERASE)

val_transforms = [
    RESIZE,
    TO_TENSOR,
]
if DO_NORMALIZE:
    # NOTE normalization is realy beneficial (created 20% accuracy difference.)
    val_transforms.append(NORMALIZE)


data_transforms = {
    "train": transforms.Compose(train_transforms),
    "val": transforms.Compose(val_transforms),
}

image_datasets = {
    d_type: datasets.ImageFolder(DATA_DIR / d_type, data_transforms[d_type]) for d_type in ["train", "val"]
}


# pin_memory=True to speed up host to device data transfer with page-locked memory
DATALOADERS = {
    d_type: torch.utils.data.DataLoader(
        image_datasets[d_type],
        batch_size=HYPS["batch_size"],
        shuffle=True,
        num_workers=HYPS["workers"],
        pin_memory=True,
    )
    for d_type in ["train", "val"]
}
DATASET_SIZES = {d_type: len(image_datasets[d_type]) for d_type in ["train", "val"]}
TRAIN_PATH = DATA_DIR / "train"
VAL_PATH = DATA_DIR / "val"


C_DICT = image_datasets["train"].class_to_idx
print(C_DICT, f"\nnumber of classes is {len(C_DICT)}")

num_samples_per_class_train = torch.unique(torch.tensor(image_datasets["train"].targets), return_counts=True)[1]
num_samples_per_class_val = torch.unique(torch.tensor(image_datasets["val"].targets), return_counts=True)[1]
CLASS_NAMES = image_datasets["train"].classes
print("CLASS_NAMES: ", CLASS_NAMES)

if TEST_AVAILABLE == True:
    # Test dataloader
    image_testset = datasets.ImageFolder(DATA_DIR / "test", data_transforms["val"])
    TEST_DATALOADER = torch.utils.data.DataLoader(
        image_testset,
        batch_size=HYPS["batch_size"],
        shuffle=True,
        num_workers=HYPS["workers"],
        pin_memory=True,
    )
    TESTSET_SIZE = len(image_testset)
    TEST_PATH = DATA_DIR / "test"
    num_samples_per_class_test = torch.unique(torch.tensor(image_testset.targets), return_counts=True)[1]
    num_samples_per_class = num_samples_per_class_train + num_samples_per_class_val + num_samples_per_class_test
    print(num_samples_per_class)  # NOTE checked the class number (They are true.)
else:
    num_samples_per_class = num_samples_per_class_train + num_samples_per_class_val
    print(num_samples_per_class)  # NOTE checked the class number (They are true.)
