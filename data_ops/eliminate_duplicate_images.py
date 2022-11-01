from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

DEF_PATH = Path(
    "/home/utku/Documents/repos/dfas_classifier/data_ops/tank_classification_dataset_combined/train/Tank-M48"
)


def filter_folder(data_folder_path: Path, deleted_images: list, images_queue):
    """Recursively iterate trough a directory to eliminate small images."""
    image_list = list(data_folder_path.iterdir())
    for img in tqdm(image_list):
        if img.is_file():
            is_deleted = compare_filter_image(img, images_queue)
            if is_deleted:
                deleted_images.append(img)


def compare_filter_image(image_file_path: Path, images_queue: list) -> bool:
    """Delete image in a path if already exists in folder."""
    is_deleted = False
    current_image = cv2.imread(str(image_file_path))
    for queue_image in images_queue:
        if current_image.shape == queue_image.shape:
            cv2_diff = cv2.subtract(current_image, queue_image)
            if not np.any(cv2_diff):
                # Images are the same, delete the image
                # image_file_path.unlink()
                is_deleted = True
    images_queue.append(current_image)
    if len(images_queue) >= 10:
        images_queue.pop(0)  # pop first element
    return is_deleted


def main(data_folder_path: Path):
    deleted_images = []
    images_queue = []
    filter_folder(data_folder_path, deleted_images, images_queue)
    with open("deleted_log", "w+") as writer:
        for del_image_path in deleted_images:
            writer.write(str(del_image_path) + "\n")
    print("Eliminated image number: ", len(deleted_images))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=DEF_PATH,
        help="",
    )
    opt = parser.parse_args()
    data_folder_path = opt.path
    main(data_folder_path)
