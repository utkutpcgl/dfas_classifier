from pathlib import Path
import torch
from torchvision import datasets, transforms
import yaml
import networkx as nx

# input image size settings
GPU_IDS = [0,1,2,3]
GPU_ID0 = GPU_IDS[0]
DEVICE = f"cuda:{GPU_ID0}"
IMG_RES_DICT = {"B0":224, "B3":300}
DATA_DIR = Path('/home/utku/Documents/repos/utkus_yolov5/classifier/create_dataset/hier_clasification_dataset/classification_dataset_combined')

# Dfas needs the graph below
DFAS_TREE = nx.DiGraph()
DFAS_TREE.add_node('araç')
DFAS_TREE.add_node('insan')
DFAS_TREE.add_edge('insan', 'askeri insan')
DFAS_TREE.add_edge('askeri', 'silahlı asker')
DFAS_TREE.add_edge('askeri', 'silahsız asker')
DFAS_TREE.add_edge('insan', 'sivil insan')
DFAS_TREE.add_edge('araç', 'sivil araç')
DFAS_TREE.add_edge('araç', 'askeri araç')
DFAS_TREE.add_edge('askeri araç', 'lastikli')
DFAS_TREE.add_edge('lastikli', 'Top-lastikli')
DFAS_TREE.add_edge('lastikli', 'lastikli araç')
DFAS_TREE.add_edge('askeri araç', 'paletli')
DFAS_TREE.add_edge('paletli', 'Tank')
DFAS_TREE.add_edge('paletli', 'ZPT')
DFAS_TREE.add_edge('paletli', 'Top-paletli')
DFAS_TREE.add_edge('Tank', 'Tank-leopard')
DFAS_TREE.add_edge('Tank', 'Tank-M60')
DFAS_TREE.add_edge('Tank', 'Tank-M48')
DFAS_TREE.add_edge('Tank', 'Tank-fırtına')

# TODO discarded "atış", "araçlı_atış", "siper_mevzi" for now
# C_DICT = {'araç': 0, 'insan': 1, 'askeri araç': 2, 'sivil araç': 3, 'askeri insan': 4, 'sivil insan': 5,
#           'lastikli': 6, 'paletli': 7, 'silahlı asker': 8, 'silahsız asker': 9, 'Top-lastikli': 10, 'lastikli araç': 11,
#           'Tank': 12, 'Top-paletli': 13, 'ZPT': 14, 'Tank-M48': 15, 'Tank-M60': 16, 'Tank-fırtına': 17, 'Tank-leopard': 18}

with open("hyperparameters.yaml","r") as reader:
    hyps = yaml.safe_load(reader)

IMG_RES = IMG_RES_DICT[hyps["model_type"]]

# TODO try without resizing and normalizing images.

data_transforms = {
    "train": transforms.Compose([
                # transforms.Resize(IMG_RES_B0),
                transforms.Resize((IMG_RES,IMG_RES)), # You must resize in effnet.
                # NOTE RandomResizedCrop was not useful
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # NOTE normalization is curcial (created 20% accuracy difference.)
            ]),
    "val": transforms.Compose([
                # Maybe apply random resized crop.
                # transforms.Resize((IMG_RES,IMG_RES)),
                transforms.Resize((IMG_RES,IMG_RES)), # You must resize in effnet.
                transforms.ToTensor(),
                # NOTE normalization is realy beneficial (created 20% accuracy difference.)
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
}

image_datasets = {
    d_type: datasets.ImageFolder(DATA_DIR/d_type, data_transforms[d_type]) for d_type in ["train", "val"]
}

# pin_memory=True to speed up host to device data transfer with page-locked memory
dataloaders = {
    d_type: torch.utils.data.DataLoader(image_datasets[d_type], batch_size = hyps["batch_size"], shuffle = True, num_workers = hyps["workers"], pin_memory=True) for d_type in ["train", "val"]
}

dataset_sizes = {d_type: len(image_datasets[d_type]) for d_type in ['train', 'val']}
class_names = image_datasets["train"].classes

# It is important that the order matches image folder order.
class_names.append("lastikli") # to avoid class error in R_constr_matrix
C_DICT = image_datasets["train"].class_to_idx
C_DICT['lastikli'] = 18 # Add final element.