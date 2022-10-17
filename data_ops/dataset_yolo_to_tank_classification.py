from math import ceil
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool
import cv2

PROC_NUM = 40  # DGX has 80 CPUS !
TARGET_YOLO_DATASET = Path()

data_splits = ["train", "val", "test"]


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


def metd(data_types, data_splits, target_yolo_dataset: Path):
    for data_split in data_splits:
        target_images_path = target_yolo_dataset / "images" / data_split
        target_labels_path = target_yolo_dataset / "labels" / data_split
        img_seq = list(img for img in target_images_path.iterdir() if img.is_file())
        img_seq.sort()
        img_partition_list = split_seq_for_porcs(img_seq, proc_num=PROC_NUM)


def get_coordinates(center_x_yolo, center_y_yolo, width_yolo, height_yolo, im_width, im_height):
    """
    Normalize boxes to convert yolo to original coordinates
    """
    min_x = (center_x_yolo - width_yolo / 2) * im_width
    min_y = (center_y_yolo - height_yolo / 2) * im_height
    max_x = (center_x_yolo + width_yolo / 2) * im_width
    max_y = (center_y_yolo + height_yolo / 2) * im_height
    return min_x, min_y, max_x, max_y


def crop_imgs(txt_file: Path, img_file: Path):
    with open("temp.txt", "r") as reader:
        str_line = reader.readline()
        while str_line:
            str_line = str_line.replace("\n", "")
            if str_line != "":
                str_elements = str_line.split(" ")
                class_idx = int(str_elements[0])
                center_x_yolo, center_y_yolo, width_yolo, height_yolo = [
                    float(str_element) for str_element in str_elements[1:]
                ]
                cv2_img = cv2.imread(img_file)
                im_height, im_width, _ = cv2_img.shape
                min_x, min_y, max_x, max_y = get_coordinates(
                    center_x_yolo, center_y_yolo, width_yolo, height_yolo, im_width, im_height
                )
                cv2_img_crop = cv2_img[min_y:max_y, min_x:max_x]  # rows, cols.
                target_crop_path = Path()
                cv2.imwrite(str(target_crop_path), cv2_img_crop)
            str_line = reader.readline()


# Added multiprocessing.
def create_yolo_annotations_mp(label_files: list, target_label_path: Path):
    """Create yolo txt label files for all waymo segmentation label files with multiprocessing.

    Args:
        label_file_segments ([str]): list containing lists of segments json file paths
        image_folder_path (Path): the image data container folder path
        label_path (Path): target label folder (contains all txt labels)
    """
    seg_partition_list = split_seq_for_porcs(label_files, proc_num=PROC_NUM)
    for seg_partition in tqdm(seg_partition_list):
        with Pool(processes=PROC_NUM) as pool:
            process_args = zip(seg_partition, repeat(target_label_path))
            pool.starmap(func=create_yolo_annotation_per_seg_file, iterable=process_args)


def create_yolo_annotation_per_seg_file(seg_label_file, target_label_path):
    with open(seg_label_file, "r") as json_file_reader:
        seg_label_json = json.load(json_file_reader)
        frames = seg_label_json["frames"]  # All the frames in a json file.
        for frame in frames:
            create_annotation_per_frame(frame, target_label_path)


def create_annotation_per_frame(frame: dict, target_label_path):
    sensors = frame["sensors"]
    # For each sensor (0-4 are cameras) there are seperate images.
    for current_sensor in sensors:
        if current_sensor["sensor_type"] != "camera":
            # Skip the lidar sensors.
            continue
        im_height = current_sensor["size_height"]
        im_width = current_sensor["size_width"]
        # Below gives the sensor specific image (front-left etc.)
        current_image_path = current_sensor["measurement_relative_path"]
        current_image_name = str(current_image_path).replace("/", "__")
        # assert (image_folder_path / current_image_path).exists(), f"image {current_image_path} does not exist."
        target_txt_path = (target_label_path / current_image_name).with_suffix(".txt")
        if target_txt_path.exists():
            # Skip current image if txt exists already.
            continue
        boxes = current_sensor["boxes"]
        box_str = ""
        for box in boxes:
            c_x, c_y, im_w, im_h, c_id = get_box_yolo(box, im_width, im_height)
            box_str = box_str + f"{c_id} {c_x} {c_y} {im_w} {im_h}\n"
        box_str = box_str[0:-1]  # NOTE remove final empty line for integrity (required by yolo)
        # Create a txt file for each image
        with open(target_txt_path, "w") as writer:
            writer.write(box_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Waymo json dataset to yolo dataset\n"
        "Read waymo json labels (segments) and convert to yolo txt files in a single folder."
    )
    parser.add_argument(
        "--source_label_path_train",
        default=f"/mnt_dgx/datasets/waymo/v1.1.1/Training/segments_json",
        help="segment path",
    )
    parser.add_argument(
        "--target_label_path_val",
        # default=f"/mnt_dgx/datasets/waymo/v1.1.1/waymo_labels/Validation",
        default=f"/raid/utku/big_datasets/waymo_labels/Validation",
        help="save path",
    )

    parser.add_argument("--mp_bool", action="store_true", help="Set if you do not want mutli processing.")
    args = parser.parse_args()
    return args


def print_args(args):
    print("\n".join(f"{k}={v}" for k, v in vars(args).items()))


def main():
    args = parse_args()
    target_label_path_val = Path(args.target_label_path_val)
    target_label_path_train = Path(args.target_label_path_train)
    target_label_path_val.mkdir(exist_ok=True, parents=True)
    target_label_path_train.mkdir(exist_ok=True, parents=True)
    print_args(args)
    if not args.mp_bool:
        print("Create yolo annotations for validation:")
        create_yolo_annotations_mp(json_label_files_val, target_label_path_val)
        print("Create yolo annotations for training:")
        create_yolo_annotations_mp(json_label_files_train, target_label_path_train)
    else:
        print("Create yolo annotations for validation:")
        create_yolo_annotations(json_label_files_val, target_label_path_val)
        print("Create yolo annotations for training:")
        create_yolo_annotations(json_label_files_train, target_label_path_train)


if __name__ == "__main__":
    main()
