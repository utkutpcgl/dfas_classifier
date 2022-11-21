from pathlib import Path
import torch
from torchvision import datasets, transforms
import yaml

# input image size settings

IMG_RES_DICT = {"B0": 224, "B3": 300}
IMG_RES_RESNET = 224
TEST_AVAILABLE = False
# TODO discarded "atış", "araçlı_atış", "siper_mevzi" for now
# C_DICT = {'araç': 0, 'insan': 1, 'askeri araç': 2, 'sivil araç': 3, 'askeri insan': 4, 'sivil insan': 5,
#           'lastikli': 6, 'paletli': 7, 'silahlı asker': 8, 'silahsız asker': 9, 'Top-lastikli': 10, 'lastikli araç': 11,
#           'Tank': 12, 'Top-paletli': 13, 'ZPT': 14, 'Tank-M48': 15, 'Tank-M60': 16, 'Tank-fırtına': 17, 'Tank-leopard': 18}
with open("hyperparameters.yaml", "r") as reader:
    HYPS = yaml.safe_load(reader)
GPU_IDS = HYPS["GPUS"]
GPU_ID0 = GPU_IDS[0]
DEVICE = f"cuda:{GPU_ID0}"

TASK = HYPS["TASK"]  # either arac or atis.
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
TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
H_FLIP = transforms.RandomHorizontalFlip()

train_transforms = [
    # NOTE RandomResizedCrop was not useful
    RESIZE,
    H_FLIP,
    TO_TENSOR,
    # NOTE normalization is curcial (created 20% accuracy difference.)
    NORMALIZE,
]
val_transforms = [
    RESIZE,
    TO_TENSOR,
    # NOTE normalization is realy beneficial (created 20% accuracy difference.)
    NORMALIZE,
]


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
