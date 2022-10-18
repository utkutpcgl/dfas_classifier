from math import ceil
from pathlib import Path
import argparse
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool
import cv2
from subprocess import run
import random

PROC_NUM = 16  # DGX has 80 CPUS !
SOURCE_YOLO_DATASET = Path(
    "/home/utku/Documents/repos/multi_label_hier_v5/data_ops/yolo_dataset_hier/yolo_dataset_all_patched_val_cleaned"
)
TARGET_CROP_DATASET = Path("/home/utku/Documents/repos/dfas_classifier/data_ops/tank_classification_dataset")
dfas_classes = (
    ["araç", "insan", "askeri araç", "sivil araç", "askeri insan"]
    + ["sivil insan", "lastikli", "paletli", "silahlı asker", "silahsız asker"]
    + ["Tank", "Top", "ZPT", "Tank-M48", "Tank-M60", "Tank-leopard"]
    + ["M110", "T155"]
)  # A total of 18 classes.
TANK_CLASS_NAMES = ["Tank-M48", "Tank-M60", "Tank-leopard"]
DATA_SPLITS = ["train", "val", "test"]


def clear_prev_dir():
    cmd = f"rm -r {TARGET_CROP_DATASET}"
    run(cmd, shell=True, check=True)


def create_dir(clear_prev=True):
    if clear_prev:
        clear_prev_dir()
    for tank_class_name in TANK_CLASS_NAMES:
        for data_split in DATA_SPLITS:
            (TARGET_CROP_DATASET / data_split / tank_class_name).mkdir(exist_ok=True, parents=True)


def split_seq_for_porcs(sequence, proc_num: int):
    """Split a list to partitions to be processed with multiprocessing starmap.

    Args:
        sequence ([str]): [target seq with depth 1 ?]
        proc_num (int): [number of processes]

    Returns:
        [list]: partition lists in a list
    """
    partition_list_len = ceil(len(sequence) / proc_num)
    partition_list = []
    for partition in range(partition_list_len):
        stat_idx = partition * proc_num
        end_idx = (partition + 1) * proc_num
        partition_list.append(sequence[stat_idx:end_idx])
    # final partition will be smaller as json_list might not be divisible to proc_num
    return partition_list


def get_coordinates(center_x_yolo, center_y_yolo, width_yolo, height_yolo, im_width, im_height):
    """
    Normalize boxes to convert yolo to original coordinates
    """
    min_x = (center_x_yolo - width_yolo / 2) * im_width
    min_y = (center_y_yolo - height_yolo / 2) * im_height
    max_x = (center_x_yolo + width_yolo / 2) * im_width
    max_y = (center_y_yolo + height_yolo / 2) * im_height
    return int(min_x), int(min_y), int(max_x), int(max_y)


def crop_single_img(img_path: Path, class_names: list, data_split: str):
    txt_path = SOURCE_YOLO_DATASET / "labels" / data_split / img_path.with_suffix(".txt").name
    if data_split == "val":
        criterion = random.randint(0, 1)
        data_split = "val" if criterion == 1 else "test"
    with open(txt_path, "r") as reader:
        str_line = reader.readline()
        count = 0
        while str_line:
            str_line = str_line.replace("\n", "")
            if str_line != "":
                str_elements = str_line.split(" ")
                class_idx = int(str_elements[0])
                class_name = dfas_classes[class_idx]
                if class_name in class_names:
                    center_x_yolo, center_y_yolo, width_yolo, height_yolo = [
                        float(str_element) for str_element in str_elements[1:]
                    ]
                    cv2_img = cv2.imread(str(img_path))
                    im_height, im_width, _ = cv2_img.shape
                    min_x, min_y, max_x, max_y = get_coordinates(
                        center_x_yolo, center_y_yolo, width_yolo, height_yolo, im_width, im_height
                    )
                    if max_x > min_x and max_y > min_y:
                        cv2_img_crop = cv2_img[min_y:max_y, min_x:max_x]  # rows, cols.
                        target_crop_path = (
                            TARGET_CROP_DATASET / data_split / class_name / f"{img_path.stem}_{count}{img_path.suffix}"
                        )
                        cv2.imwrite(str(target_crop_path), cv2_img_crop)
                        count += 1
            str_line = reader.readline()


def crop_tanks_in_imgs_mp():
    """Save tank crops from images given yolo labels with multiprocessing.
    label_file_segments ([str]): list containing lists of segments json file paths
    """
    for data_split in DATA_SPLITS:
        source_images_path = SOURCE_YOLO_DATASET / "images" / data_split
        img_seq = list(img for img in source_images_path.iterdir() if img.is_file())
        img_seq.sort()
        img_partition_list = split_seq_for_porcs(img_seq, proc_num=PROC_NUM)
        for img_partition in tqdm(img_partition_list):
            with Pool(processes=PROC_NUM) as pool:
                process_args = zip(img_partition, repeat(TANK_CLASS_NAMES), repeat(data_split))
                pool.starmap(func=crop_single_img, iterable=process_args)


def crop_tanks_in_imgs():
    """Save tank crops from images given yolo labels.
    label_file_segments ([str]): list containing lists of segments json file paths
    """
    for data_split in DATA_SPLITS:
        source_images_path = SOURCE_YOLO_DATASET / "images" / data_split
        img_seq = list(img for img in source_images_path.iterdir() if img.is_file())
        img_seq.sort()
        for img_path in tqdm(img_seq):
            crop_single_img(img_path, class_names=TANK_CLASS_NAMES, data_split=data_split)


def parse_args():
    parser = argparse.ArgumentParser(description="Yolo dataset to cropped tank classification dataset.")
    parser.add_argument("--mp_bool", action="store_true", help="Set if you do not want mutli processing.")
    args = parser.parse_args()
    return args


def print_args(args):
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))


def main():
    create_dir()
    args = parse_args()
    print_args(args)
    if not args.mp_bool:
        print("Create cropped classification dataset WITHOUT multiproc.")
        crop_tanks_in_imgs()
    else:
        print("Create cropped classification dataset WITH multiproc.")
        crop_tanks_in_imgs_mp()


if __name__ == "__main__":
    main()
