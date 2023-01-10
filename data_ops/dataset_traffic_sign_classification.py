import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import subprocess
from matplotlib import pyplot as plt
from tqdm import tqdm
from icecream import ic


class_names = [
 "warning--traffic-signals--g1",
 "warning--roadworks",
 "warning--road-bump--g1",
 "warning--other-danger--g1",
 "warning--children--g1",
 "stop_sign",
 "regulatory--yield--g1",
 "regulatory--no-u-turn--g1",
 "regulatory--no-right-turn--g1",
 "regulatory--no-parking--g1",
 "regulatory--no-overtaking--g1",
 "regulatory--no-left-turn--g1",
 "regulatory--no-entry--g1",
 "regulatory--maximum-speed-limit-90--g1",
 "regulatory--maximum-speed-limit-80--g1",
 "regulatory--maximum-speed-limit-70--g1",
 "regulatory--maximum-speed-limit-60--g1",
 "regulatory--maximum-speed-limit-50--g1",
 "regulatory--maximum-speed-limit-40--g1",
 "regulatory--maximum-speed-limit-30--g1",
 "regulatory--maximum-speed-limit-20--g1",
 "regulatory--maximum-speed-limit-120--g1",
 "regulatory--maximum-speed-limit-110--g1",
 "regulatory--maximum-speed-limit-100--g1",
 "regulatory--maximum-speed-limit-10--g1",
 "pedestrians-crossing--g1",
 "other",
]


def create_directories(data_parser):
    # CREATE directories
    for class_name in class_names:
        (data_parser.raw_all_dataset / class_name).mkdir(exist_ok=True, parents=True)
    # for split_type in ["train", "val", "test"]:
    #     (data_parser.classification_dataset / split_type).mkdir(exist_ok=True)
    #     for class_name in class_names:
    #         (data_parser.classification_dataset / split_type / class_name).mkdir(exist_ok=True)


def get_sorted_xml_files_paths(dataset_folder_paths: list):
    xml_files = []
    for path in dataset_folder_paths:
        if path.suffix == ".xml":
            xml_files.append(path)
    xml_files.sort()
    return xml_files


def clear_previous_files(data_parser):
    if data_parser.classification_dataset.exists():
        cmd = f"rm -r {data_parser.classification_dataset}"
        subprocess.run(cmd, shell=True, check=True)


class DataParserTS:
    def __init__(self, dataset_folder=Path, filter_bbox: bool = False, clear_prev_files: bool = False):
        self.filter_bbox = filter_bbox
        self.dataset_folder = dataset_folder
        self.classification_dataset = Path("traffic_sign_clasification_dataset/clasification_dataset_cerkez")
        self.raw_all_dataset = self.classification_dataset / "all"
        if clear_prev_files:
            # Remove previous dataset directory:
            clear_previous_files(self)
        create_directories(self)

    def process(self, xmlfiles: list, videofiles: list) -> None:
        """Process previous cerkez labels to match new dfas label style."""
        total_bbox_count = 0
        length = len(xmlfiles)
        for k in range(length):
            video = videofiles[k]
            xmlfile = xmlfiles[k]
            print(f"{xmlfile}, is {k}th video out of {length} videos")
            scene_name = Path(video).stem
            cap = cv2.VideoCapture(str(video))
            xml_tree = ET.parse(xmlfile)
            root = xml_tree.getroot()
            images_list = list(image for image in root.iter("image"))
            total_images = len(images_list)
            ic(total_images)
            frame_index = 0
            while cap.isOpened():
                retval, frame = cap.read()
                if not retval:
                    break
                print(f"DataParserCerkez progress [%d/{total_images}] \r" % frame_index, end="")
                img = images_list[frame_index]
                frame_index += 1
                for box in img.iter("box"):
                    # label extraction.
                    label = box.get("label")
                    xtl = int(round(float(box.get("xtl"))))
                    if xtl < 0:
                        xtl = 0
                    ytl = int(round(float(box.get("ytl"))))
                    xbr = int(round(float(box.get("xbr"))))
                    ybr = int(round(float(box.get("ybr"))))

                    img_path = (
                        self.raw_all_dataset
                        / label
                        / f"{scene_name}_{frame_index}_{str(total_bbox_count)}.jpg"
                    )
                    cropped_image = frame[ytl:ybr, xtl:xbr, :]
                    cv2.imwrite(str(img_path), cropped_image)
                    total_bbox_count += 1
                # Write image specific labels to yolo label txt

    def parse(self):
        # Get xml files and image folders.
        xml_file_paths = list(folder for folder in self.dataset_folder.iterdir() if folder.suffix == ".xml")
        dataset_folder_paths_cerkez = list(folder for folder in self.dataset_folder.iterdir() if folder.is_dir())
        xml_files_cerkez = get_sorted_xml_files_paths(xml_file_paths)
        assert len(xml_files_cerkez) == len(
            dataset_folder_paths_cerkez
        ), "# of xml files is not equal to # of image folders in the path."
        video_files = []
        for folder in dataset_folder_paths_cerkez:
            if folder.is_dir():
                video_path = (folder / folder.stem).with_suffix(".mp4")
                video_files.append(video_path)
        video_files.sort()
        ic(video_files)
        ic(xml_files_cerkez)
        self.process(xml_files_cerkez, video_files)


if __name__ == "__main__":
    dataset_folder_traffic_sign = Path("/home/utku/Documents/raw_datasets/Traffic_sign_light_classification/VK_datasets/apems_vehicle_radar_tests")

    parser_traffic_sign = ic(DataParserTS(dataset_folder=dataset_folder_traffic_sign, filter_bbox=False))
    ic(parser_traffic_sign.parse())
    # ic(split_dataset(parser_traffic_sign, scene_name="cerkez"))
