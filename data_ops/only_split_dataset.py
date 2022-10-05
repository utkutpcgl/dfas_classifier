from os import mkdir
import random
import cv2
from pathlib import Path
import shutil
import subprocess
from matplotlib import pyplot as plt
from tqdm import tqdm
from icecream import ic
from multiprocessing import Pool, Value, Lock
import math
from itertools import repeat


PROC_NUM = 8
total_bbox_count_mp = Value("i", 0)
occluded_count_mp = Value("i", 0)
occluded_count = 0  # global variable.
lock = Lock()
SKIP_OCCLUDED = True
TRAIN_RATIO = 75 / 100

ALL_DATASET = Path("/home/utku/Documents/repos/dfas_classifier/data_ops/arac_yönelim_classification_dataset/all")
MIXED_ALL_DATASET = Path(
    "/home/utku/Documents/repos/dfas_classifier/data_ops/arac_yönelim_classification_dataset/mixed_all"
)

TARGET_CLASSIFICATION_DATASET = Path(
    "/home/utku/Documents/repos/dfas_classifier/data_ops/arac_yönelim_classification_dataset/arac_yonelim_classification_dataset_combined"
)


def create_dir():
    if TARGET_CLASSIFICATION_DATASET.exists():
        cmd = f"rm -r {TARGET_CLASSIFICATION_DATASET}"
        subprocess.run(cmd, check=True, shell=True)
    for split in ["test", "train", "val"]:
        for class_name in ["dik", "paralel"]:
            (TARGET_CLASSIFICATION_DATASET / split / class_name).mkdir(exist_ok=True, parents=True)


# Move all images to single folder (names will include class names at the end.)
def copy_images_mixed(all_dataset: Path, mixed_all_dataset: Path):
    if Path(MIXED_ALL_DATASET).exists():
        cmd = f"rm -r {MIXED_ALL_DATASET}"
        subprocess.run(cmd, check=True, shell=True)
    MIXED_ALL_DATASET.mkdir(exist_ok=True)
    for class_folder in all_dataset.iterdir():
        class_name = class_folder.name
        for image in class_folder.iterdir():
            target_path = mixed_all_dataset / (image.stem + f"_{class_name}" + image.suffix)
            shutil.copyfile(image, target_path)


# Rename files to have a proper sorted order for seperating val and train sets task specific.
def pad_zeros(num_str, max_elements):
    pad_this_many = max_elements - len(num_str)
    new_num_str = pad_this_many * "0" + num_str
    return new_num_str


def get_new_file(image_name):
    split_image_name_list = image_name.split("_")
    # index -2 and -3 should be padded 0s (max 6 elements)
    split_image_name_list[-2] = pad_zeros(split_image_name_list[-2], 6)
    split_image_name_list[-3] = pad_zeros(split_image_name_list[-3], 6)
    new_image_name = "_".join(split_image_name_list)
    return new_image_name


def rename_files_to_sort(mixed_all_dataset):
    for image_file in mixed_all_dataset.iterdir():
        image_file.rename(image_file.parent / get_new_file(image_file.name))


# Based on the tasks find the train indexes vs val test indexes.
def extract_task_and_class_name(image_name):
    split_image_name_list = image_name.split("_")
    task_name = "_".join(split_image_name_list[:-3])
    class_name = split_image_name_list[-1].split(".")[0]
    return task_name, class_name


def create_val_train_split_lists(images_path_list: Path, train_ratio=TRAIN_RATIO):
    """Create lists that split all tasks to train and validation."""
    train_image_path_list = []
    val_image_path_list = []
    prev_task_name = ""
    task_based_image_path_lists = []
    for image_path in images_path_list:
        if image_path.is_file():
            task_name, class_name = extract_task_and_class_name(image_path.name)
            if prev_task_name == task_name:
                task_based_image_path_lists[-1].append((image_path, class_name))
            else:
                task_based_image_path_lists.append([(image_path, class_name)])
            prev_task_name = task_name
    for task_image_path_list in task_based_image_path_lists:
        # Split the list to train and validation.
        len_of_task_image_list = len(task_image_path_list)
        train_image_count = int(len_of_task_image_list * train_ratio)
        train_image_path_list.extend(task_image_path_list[0:train_image_count])
        val_image_path_list.extend(task_image_path_list[train_image_count:])
    return train_image_path_list, val_image_path_list


# Then split images to data_splits (train,val,test) and create the class folders.
def plot_bar_graph_per_split(number_per_split: dict, plot_name: str):
    splits_keys = list(number_per_split.keys())
    values = list(number_per_split.values())
    plt.figure(figsize=(10, 7))
    plt.barh(splits_keys, values)
    plt.xlabel("Split names")
    plt.ylabel("No of elements")
    plt.title(plot_name)
    plt.savefig(plot_name)
    plt.clf()


def split_dataset(
    target_classification_dataset,
    mixed_all_dataset,
    train_ratio=TRAIN_RATIO,
    create_test_split=True,
    scene_name="arac_yonelim",
):
    cmd = f"find {mixed_all_dataset} -type f | wc -l"
    num_of_files_stdout = subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE)
    number_of_images = int(str(num_of_files_stdout.stdout).replace("\\n", "").replace("b", "").replace("'", ""))
    ic("Number of images for classification in dfas dataset folder: ", number_of_images)
    train_end_index = int(number_of_images * train_ratio)
    ic("Number of images training images: ", train_end_index)
    number_per_classes_per_split = {
        data_split: {class_name: 0 for class_name in ["dik", "paralel"]} for data_split in ["train", "val", "test"]
    }
    image_path_list = list(image_path for image_path in mixed_all_dataset.iterdir() if image_path.is_file())
    image_path_list.sort()
    train_image_path_list, val_image_path_list = create_val_train_split_lists(image_path_list, train_ratio)
    # 0.75, 0.125, 0.125 -> train,val,test split
    for image, class_name in tqdm(train_image_path_list):
        target_image_path = target_classification_dataset / "train" / class_name / image.name
        shutil.copyfile(image, target_image_path)
        number_per_classes_per_split["train"][class_name] += 1

    for image, class_name in tqdm(val_image_path_list):
        criterion = random.randint(0, 1)
        if not create_test_split or criterion == 0:
            # val
            target_image_path = target_classification_dataset / "val" / class_name / image.name
            number_per_classes_per_split["val"][class_name] += 1
        else:
            # test criterion == 1
            target_image_path = target_classification_dataset / "test" / class_name / image.name
            number_per_classes_per_split["test"][class_name] += 1
        shutil.copyfile(image, target_image_path)
    for split_name, classes_per_split in number_per_classes_per_split.items():
        plot_bar_graph_per_split(classes_per_split, plot_name=f"{scene_name}_{split_name}")


def main():
    create_dir()
    copy_images_mixed(all_dataset=ALL_DATASET, mixed_all_dataset=MIXED_ALL_DATASET)
    rename_files_to_sort(mixed_all_dataset=MIXED_ALL_DATASET)
    split_dataset(target_classification_dataset=TARGET_CLASSIFICATION_DATASET, mixed_all_dataset=MIXED_ALL_DATASET)


if __name__ == "__main__":
    main()
