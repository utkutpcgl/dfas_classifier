from pathlib import Path
from argparse import ArgumentParser
from PIL import Image

WIDTH_LIMIT = 35
HEIGHT_LIMIT = 25
DEF_PATH = Path("/home/utku/Documents/repos/dfas_classifier/create_dataset/atis_yönelim_clasification_dataset")


def filter_folder_recursively(data_folder_path: Path, deleted_images: list):
    """Recursively iterate trough a directory to eliminate small images."""
    for folder_element in data_folder_path.iterdir():
        if folder_element.is_file():
            is_deleted = filter_image_wrt_size(folder_element)
            if is_deleted:
                deleted_images.append(folder_element)
        else:
            # Iterate over the directory (only on train,val,test)
            if folder_element.name != "all":
                filter_folder_recursively(folder_element, deleted_images)


def filter_image_wrt_size(image_file_path: Path) -> bool:
    """Delete image in a path if smaller than desired.
    Returns true if deleted else false."""
    is_deleted = False
    image = Image.open(image_file_path)
    width, height = image.size
    if width < WIDTH_LIMIT or height < HEIGHT_LIMIT:
        # Delete the image
        print("enters here")
        image_file_path.unlink()
        is_deleted = True
    return is_deleted


def main(data_folder_path: Path):
    deleted_images = []
    filter_folder_recursively(data_folder_path, deleted_images)
    with open("deleted_log", "w") as writer:
        for del_image_path in deleted_images:
            writer.write(str(del_image_path))
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
