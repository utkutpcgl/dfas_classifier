from imp import lock_held
import random
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from networkx.algorithms.dag import ancestors
import networkx as nx
import shutil
import subprocess
from matplotlib import pyplot as plt
from sklearn.neighbors import VALID_METRICS
from sklearn.preprocessing import scale
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

"""
Create classification dataset for torchvision. Folders should be named after classes in train/val/test folders.
"""

# TODO LIST
# Take image folder and split labels and images to combined dataset from different xml files and image folders of sepearate tasks.
#

# Dfas needs the graph below
dfas_tree = nx.DiGraph()
dfas_tree.add_node("araç")
dfas_tree.add_node("insan")
dfas_tree.add_edge("insan", "askeri insan")
dfas_tree.add_edge("askeri", "silahlı asker")
dfas_tree.add_edge("askeri", "silahsız asker")
dfas_tree.add_edge("insan", "sivil insan")
dfas_tree.add_edge("araç", "sivil araç")
dfas_tree.add_edge("araç", "askeri araç")
dfas_tree.add_edge("askeri araç", "lastikli")
dfas_tree.add_edge("lastikli", "Top-lastikli")
dfas_tree.add_edge("lastikli", "lastikli araç")
dfas_tree.add_edge("askeri araç", "paletli")
dfas_tree.add_edge("paletli", "Tank")
dfas_tree.add_edge("paletli", "ZPT")
dfas_tree.add_edge("paletli", "Top-paletli")
dfas_tree.add_edge("Tank", "Tank-leopard")
dfas_tree.add_edge("Tank", "Tank-M60")
dfas_tree.add_edge("Tank", "Tank-M48")
dfas_tree.add_edge("Tank", "Tank-fırtına")

# NOTE we care about tank, m60, m48, leopard
dfas_tanks = ["Tank-leopard", "Tank-M60", "Tank-M48", "Tank-fırtına", "Tank"]
dfas_lut = {"belirsiz": "dogrultmamis", "doğrultmuş": "dogrultmus", "doğrultmamış": "dogrultmamis"}

# Cerkezköy needs the graph below
cerkez_tree = nx.DiGraph()
cerkez_tree.add_node("insan")
cerkez_tree.add_node("arac")
cerkez_tree.add_edge("insan", "askeri")
cerkez_tree.add_edge("askeri", "silahlı")
cerkez_tree.add_edge("askeri", "silahsız")
cerkez_tree.add_edge("insan", "sivil")
cerkez_tree.add_edge("arac", "sivil arac")
cerkez_tree.add_edge("arac", "askeri arac")
cerkez_tree.add_edge("askeri arac", "lastikli")
cerkez_tree.add_edge("askeri arac", "paletli")
cerkez_tree.add_edge("paletli", "tank")
cerkez_tree.add_edge("paletli", "ZMA")
cerkez_tree.add_edge("tank", "leopard")
cerkez_tree.add_edge("tank", "m60")
cerkez_tree.add_edge("tank", "m48")

# NOTE we care about tank, m60, m48, leopard
cerkez_tanks = ["tank", "m60", "m48", "leopard"]
cerkez_lut = {"yok": "dogrultmamis", "dogrultmus": "dogrultmus", "dogrultmamis": "dogrultmamis"}

# attribute is silah_durumu for dfas and silah durum for cerkez
class_names = {"dogrultmus", "dogrultmamis"}


def create_directories(data_parser):
    # CREATE directories
    for class_name in class_names:
        (data_parser.raw_all_dataset / class_name).mkdir(exist_ok=True, parents=True)
    for split_type in ["train", "val", "test"]:
        (data_parser.classification_dataset / split_type).mkdir(exist_ok=True)
        for class_name in class_names:
            (data_parser.classification_dataset / split_type / class_name).mkdir(exist_ok=True)


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


def split_dataset(data_parser, train_ratio=4 / 5, old_data_split=False, scene_name: str = "dfas"):
    """Hardcoded dataset split."""

    def plot_bar_graph_per_split(classes_per_split: dict, plot_name: str):
        classes = list(classes_per_split.keys())
        values = list(classes_per_split.values())
        plt.figure(figsize=(10, 7))
        plt.barh(classes, values)
        plt.xlabel("Classes")
        plt.ylabel("No of elements")
        plt.title(plot_name)
        plt.savefig(plot_name)
        plt.clf()

    # 0.75, 0.125, 0.125 -> train,val,test split
    cmd = f"find {data_parser.raw_all_dataset} -type f | wc -l"
    num_of_files_stdout = subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE)
    number_of_images = int(str(num_of_files_stdout.stdout).replace("\\n", "").replace("b", "").replace("'", ""))
    ic("Number of images for classification in dfas dataset folder: ", number_of_images)
    train_end_index = int(number_of_images * train_ratio)
    ic("Number of images training images: ", train_end_index)
    number_per_classes_per_split = {
        data_split: {class_name: 0 for class_name in class_names} for data_split in ["train", "val", "test"]
    }
    for class_folder in data_parser.raw_all_dataset.iterdir():
        ic(class_folder)
        enumerated_class_folder_list = list(enumerate(list(class_folder.iterdir())))
        class_folder_len = len(enumerated_class_folder_list)
        class_train_end_index = int(class_folder_len * train_ratio)
        for image_index, image_in_class in tqdm(enumerated_class_folder_list):
            if image_in_class.is_file():
                if old_data_split:
                    with open("atis_test_image_names.txt", "r") as test_images_reader:
                        test_image_names = test_images_reader.read().split("\n")
                        test_image_names.remove("")
                    with open("atis_val_image_names.txt", "r") as val_images_reader:
                        val_image_names = val_images_reader.read().split("\n")
                        val_image_names.remove("")
                    if image_in_class.name in test_image_names:
                        data_split = "test"
                    elif image_in_class.name in val_image_names:
                        data_split = "val"
                    else:
                        data_split = "train"
                else:
                    criterion = random.randint(0, 1)
                    if image_index <= class_train_end_index:
                        data_split = "train"
                    else:
                        if criterion == 0:
                            data_split = "val"
                        elif criterion == 1:
                            data_split = "test"
                number_per_classes_per_split[data_split][class_folder.name] += 1
                target_image_path = (
                    (data_parser.classification_dataset / data_split) / class_folder.name / image_in_class.name
                )
                shutil.copyfile(image_in_class, target_image_path)
    ic(number_per_classes_per_split)
    for split_name, classes_per_split in number_per_classes_per_split.items():
        plot_bar_graph_per_split(classes_per_split, plot_name=f"{scene_name}_{split_name}")


# Skip the label if the width and height are not appropriate
def check_filter_bboxes(label, bbox_width, bbox_height, car_h_thr=26, car_wh_thr=18, person_h_thr=20, person_wh_thr=10):
    words_in_label = label.split(" ")
    skip_bbox = False
    if "insan" in words_in_label or "asker" in words_in_label:
        # Filter for person
        if person_wh_thr > bbox_width or person_h_thr > bbox_height:
            skip_bbox = True
    else:
        # Filter for car
        if car_wh_thr > bbox_width or car_h_thr > bbox_height:
            skip_bbox = True
    return skip_bbox


def combine_datasets(dataset_list, target_dataset=Path("classification_dataset_combined")):
    target_dataset.mkdir(exist_ok=True)
    for dataset in dataset_list:
        # Copy with update, adds missing images to matching folders.
        cmd = f"cp -ur {dataset}/* {target_dataset}/"
        subprocess.run(cmd, check=True, shell=True)


class DataParserDFAS:
    def __init__(
        self,
        dataset_folder: Path,
        out_classification_dataset: str = None,
        filter_bbox: bool = False,
        clear_prev_files: bool = True,
    ):
        self.filter_bbox = filter_bbox
        self.dataset_folder = dataset_folder
        if out_classification_dataset != None:
            self.classification_dataset = Path(out_classification_dataset)
        else:
            self.classification_dataset = Path("atis_yönelim_classification_dataset/classification_dataset_dfas")
        self.raw_all_dataset = self.classification_dataset / "all"
        if clear_prev_files:
            # Remove previous dataset directory:
            clear_previous_files(self)
        create_directories(self)

        # Get xml files and image folders.
        xml_file_paths = list(folder for folder in self.dataset_folder.iterdir() if folder.suffix == ".xml")
        self.dataset_folder_paths = list(folder for folder in self.dataset_folder.iterdir() if folder.is_dir())
        self.xml_files = get_sorted_xml_files_paths(xml_file_paths)
        assert len(self.xml_files) == len(
            self.dataset_folder_paths
        ), f"# of xml files {len(self.xml_files)} is not equal to # of image folders {len(self.dataset_folder_paths)} in the path."

        self.image_folders = self.get_sorted_image_folder_paths(self.dataset_folder_paths, self.xml_files)
        assert len(self.xml_files) == len(
            self.image_folders
        ), "# of xml files is not equal to # of image folders after choosing only available label folders."
        self.label_and_image_paths = list(zip(self.xml_files, self.image_folders))
        ic(f"There are {len(self.label_and_image_paths)} labelled image folders available.")

    def get_sorted_image_folder_paths(self, dataset_folder_paths: list, xml_files: list):
        image_folders = []
        # Only add image folder path if the corresponding label file exists.
        for path in dataset_folder_paths:
            if path.suffix != ".xml":
                for xml_file in xml_files:
                    if path.stem in str(xml_file):
                        image_folders.append(path)
        image_folders.sort()
        return image_folders

    def proc_initializer(self, *args):
        global lock, total_bbox_count_mp, occluded_count_mp
        total_bbox_count_mp, occluded_count_mp, lock = args

    def process(self, image_props: list, multi_proc: bool):
        total_bbox_count = 0
        for xmlfile, image_folder in self.label_and_image_paths:
            image_height = image_props[xmlfile]["height"]
            image_width = image_props[xmlfile]["width"]
            image_path = image_props[xmlfile]["path"]
            scene_name = Path(image_folder).stem.split("--")[0]  # get only the name
            ic(xmlfile)
            ic()
            xml_tree = ET.parse(xmlfile)
            root = xml_tree.getroot()
            all_images = root.findall("image")

            enumerated_all_images = list(enumerate(all_images))
            if multi_proc:
                number_of_image_packets = math.ceil(len(enumerated_all_images) / PROC_NUM)
                with Pool(
                    processes=PROC_NUM,
                    initializer=self.proc_initializer,
                    initargs=(total_bbox_count_mp, occluded_count_mp, lock),
                ) as pool:
                    for idx in tqdm(range(number_of_image_packets)):
                        start_idx = idx * PROC_NUM
                        enumerated_image_list = enumerated_all_images[start_idx : start_idx + PROC_NUM]
                        process_args = zip(
                            enumerated_image_list,
                            repeat(image_height),
                            repeat(image_width),
                            repeat(scene_name),
                            repeat(image_folder),
                            repeat(total_bbox_count),
                            repeat(True),
                            repeat(xmlfile),
                        )
                        pool.starmap(func=self.process_image, iterable=process_args)
                    if SKIP_OCCLUDED:
                        ic(f"Skipped {occluded_count_mp.value} occluded images in {xmlfile}")
                        occluded_count_mp.value = 0
            else:
                for frame_index_img_tuple in tqdm(enumerated_all_images):
                    total_bbox_count = self.process_image(
                        frame_index_img_tuple=frame_index_img_tuple,
                        image_height=image_height,
                        image_width=image_width,
                        scene_name=scene_name,
                        image_folder=image_folder,
                        total_bbox_count=total_bbox_count,
                        multi_proc=False,
                        xmlfile=xmlfile,
                    )
                if SKIP_OCCLUDED:
                    global occluded_count
                    ic(f"Skipped {occluded_count} occluded images in {xmlfile}")
                    occluded_count = 0

    # def process_mp(self, image_props: list):
    #     multi_proc = True
    #     total_bbox_count = Value("i", 0)  # has to be multiproc value in multiprocessing.
    #     for xmlfile, image_folder in self.label_and_image_paths:
    #         image_height = image_props[xmlfile]["height"]
    #         image_width = image_props[xmlfile]["width"]
    #         image_path = image_props[xmlfile]["path"]
    #         scene_name = Path(image_folder).stem.split("--")[0]  # get only the name
    #         ic(xmlfile)
    #         ic()
    #         xml_tree = ET.parse(xmlfile)
    #         root = xml_tree.getroot()
    #         all_images = root.findall("image")

    #         enumerated_all_images = list(enumerate(all_images))
    #         number_of_image_packets = math.ceil(len(enumerated_all_images) / PROC_NUM)
    #         with Pool(
    #             processes=PROC_NUM, initializer=self.proc_initializer, initargs=(total_bbox_count_mp, lock)
    #         ) as pool:
    #             for idx in tqdm(range(number_of_image_packets)):
    #                 start_idx = idx * PROC_NUM
    #                 enumerated_image_list = enumerated_all_images[start_idx : start_idx + PROC_NUM]
    #                 process_args = zip(
    #                     enumerated_image_list,
    #                     repeat(image_height),
    #                     repeat(image_width),
    #                     repeat(scene_name),
    #                     repeat(image_folder),
    #                 )
    #                 pool.starmap(func=self.process_image_mp, iterable=process_args)

    def process_image(
        self,
        frame_index_img_tuple,
        image_height,
        image_width,
        scene_name,
        image_folder,
        total_bbox_count,
        multi_proc,
        xmlfile,
    ):
        frame_index, img = frame_index_img_tuple
        # img = next(images)
        img_name = img.get("name")
        # TODO check if the image width height is the same in the xml annotation file and scale accordingly.
        xml_img_width = int(img.get("width"))
        xml_img_height = int(img.get("height"))
        assert (xml_img_height / xml_img_width) == (
            image_height / image_width
        ), f"{xml_img_height / xml_img_width}!={image_height  / image_width}"
        scale_pixels = 1
        if xml_img_width != image_width:
            scale_pixels = image_width / xml_img_width
        image_path = image_folder / f"{scene_name}--{img_name}"
        cv2_image = cv2.imread(str(image_path))
        for box in img.iter("box"):
            self.crop_box(
                box,
                scale_pixels,
                scene_name,
                frame_index,
                cv2_image,
                total_bbox_count,
                multiproc=multi_proc,
                xmlfile=xmlfile,
            )
        return total_bbox_count

    # def process_image_mp(self, frame_index_img_tuple, image_height, image_width, scene_name, image_folder, xmlfile):
    #     frame_index, img = frame_index_img_tuple
    #     # img = next(images)
    #     img_name = img.get("name")
    #     # TODO check if the image width height is the same in the xml annotation file and scale accordingly.
    #     xml_img_width = int(img.get("width"))
    #     xml_img_height = int(img.get("height"))
    #     assert (xml_img_height / xml_img_width) == (
    #         image_height / image_width
    #     ), f"{xml_img_height / xml_img_width}!={image_height  / image_width}"
    #     scale_pixels = 1
    #     if xml_img_width != image_width:
    #         scale_pixels = image_width / xml_img_width
    #     image_path = image_folder / f"{scene_name}--{img_name}"
    #     cv2_image = cv2.imread(str(image_path))
    #     for box in img.iter("box"):
    #         self.crop_box_mp(box, scale_pixels, scene_name, frame_index, cv2_image)

    # def crop_box_mp(self, box, scale_pixels, scene_name, frame_index, cv2_image):
    #     # label extraction.
    #     # includes all classes (without doğrultmuş/doğrultmamış ve lastik_palet_izi.)
    #     label = box.get("label")
    #     # We only care about tanks currently.
    #     if label not in dfas_tanks:
    #         return

    #     box_dict = {}
    #     for attr in box.iter("attribute"):
    #         attr_name = attr.get("name")
    #         answer = attr.text
    #         box_dict[attr_name] = answer
    #     atis_yönelim = box_dict["silah_durumu"]  # silah_durumu is the attribute name.
    #     atis_yönelim_label = dfas_lut[atis_yönelim]

    #     # Scale if the annotation file does not have the correct image resolution.
    #     xtl = int(round(float(box.get("xtl"))) * scale_pixels)
    #     if xtl < 0:
    #         xtl = 0
    #     ytl = int(round(float(box.get("ytl"))) * scale_pixels)
    #     xbr = int(round(float(box.get("xbr"))) * scale_pixels)
    #     ybr = int(round(float(box.get("ybr"))) * scale_pixels)

    #     # for output vector
    #     # YOLO label conversion
    #     # Normalize the pixel coordinates for yolo
    #     bbox_width = xbr - xtl
    #     bbox_height = ybr - ytl
    #     # Skip bbox if too small.
    #     if self.filter_bbox:
    #         skip_bbox = check_filter_bboxes(atis_yönelim_label, bbox_width=bbox_width, bbox_height=bbox_height)
    #         if skip_bbox:
    #             return
    #     with lock:
    #         # lock the total_bbox_count value to avoid race conditions in multiprocessing.
    #         img_path = (
    #             self.raw_all_dataset
    #             / atis_yönelim_label
    #             / f"dfas_{scene_name}_{frame_index}_{str(total_bbox_count_mp.value)}.jpg"
    #         )
    #         total_bbox_count_mp.value += 1
    #     cropped_image = cv2_image[ytl:ybr, xtl:xbr, :]
    #     cv2.imwrite(str(img_path), cropped_image)

    def crop_box(self, box, scale_pixels, scene_name, frame_index, cv2_image, total_bbox_count, multiproc, xmlfile):
        if SKIP_OCCLUDED:
            global occluded_count
            if "sereflikochisar_3" in str(xmlfile):
                is_occluded = int(box.get("occluded"))
                if is_occluded == 1:
                    if multiproc:
                        occluded_count_mp.value += 1
                    else:
                        occluded_count += 1
                    return total_bbox_count
        # label extraction.
        # includes all classes (without doğrultmuş/doğrultmamış ve lastik_palet_izi.)
        label = box.get("label")
        # We only care about tanks currently.
        if label not in dfas_tanks:
            return total_bbox_count

        box_dict = {}
        for attr in box.iter("attribute"):
            attr_name = attr.get("name")
            answer = attr.text
            box_dict[attr_name] = answer
        atis_yönelim = box_dict["silah_durumu"]  # silah_durumu is the attribute name.
        atis_yönelim_label = dfas_lut[atis_yönelim]

        # Scale if the annotation file does not have the correct image resolution.
        xtl = int(round(float(box.get("xtl"))) * scale_pixels)
        if xtl < 0:
            xtl = 0
        ytl = int(round(float(box.get("ytl"))) * scale_pixels)
        xbr = int(round(float(box.get("xbr"))) * scale_pixels)
        ybr = int(round(float(box.get("ybr"))) * scale_pixels)

        # for output vector
        # YOLO label conversion
        # Normalize the pixel coordinates for yolo
        bbox_width = xbr - xtl
        bbox_height = ybr - ytl

        # Skip bbox if too small.
        if self.filter_bbox:
            skip_bbox = check_filter_bboxes(atis_yönelim_label, bbox_width=bbox_width, bbox_height=bbox_height)
            if skip_bbox:
                return

        if multiproc:
            with lock:
                # lock the total_bbox_count value to avoid race conditions in multiprocessing.
                img_path = (
                    self.raw_all_dataset
                    / atis_yönelim_label
                    / f"dfas_{scene_name}_{frame_index}_{str(total_bbox_count_mp.value)}.jpg"
                )
                total_bbox_count_mp.value += 1
        else:
            img_path = (
                self.raw_all_dataset
                / atis_yönelim_label
                / f"dfas_{scene_name}_{frame_index}_{str(total_bbox_count)}.jpg"
            )
            total_bbox_count += 1

        cropped_image = cv2_image[ytl:ybr, xtl:xbr, :]
        cv2.imwrite(str(img_path), cropped_image)
        return total_bbox_count  # Return value only meaningful if multiproc==False.

    def parse(self, multi_proc=False):
        img_dims_dict = {}
        for xml_file, image_folder_path in self.label_and_image_paths:
            image_folder_list = list(im for im in image_folder_path.iterdir() if im.suffix in [".png", ".jpg", ".jpeg"])
            an_image_path_in_folder = image_folder_list[0]
            ic(an_image_path_in_folder)
            temp_img = cv2.imread(str(an_image_path_in_folder))
            height, width, channels = temp_img.shape
            img_dims_dict[xml_file] = {"width": width, "height": height, "path": image_folder_path}
        self.process(img_dims_dict, multi_proc=multi_proc)


class DataParserCerkez:
    def __init__(self, dataset_folder=Path, filter_bbox: bool = False, clear_prev_files: bool = False):
        self.filter_bbox = filter_bbox
        self.dataset_folder = dataset_folder
        self.classification_dataset = Path("atis_yönelim_classification_dataset/classification_dataset_cerkez")
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
                    box_dict = {}
                    if box.get("label") != "arac":  # skip if not araç
                        continue
                    for attr in box.iter("attribute"):
                        attr_name = attr.get("name")
                        answer = attr.text
                        box_dict[attr_name] = answer
                    most_specific_label = box_dict["ek nitelik"]
                    if most_specific_label not in cerkez_tanks:
                        continue
                    atis_yönelim = box_dict["silah durum"]
                    atis_yönelim_label = cerkez_lut[atis_yönelim]
                    # top left and bottom right label_list are available.
                    xtl = int(round(float(box.get("xtl"))))
                    if xtl < 0:
                        xtl = 0
                    ytl = int(round(float(box.get("ytl"))))
                    xbr = int(round(float(box.get("xbr"))))
                    ybr = int(round(float(box.get("ybr"))))

                    # Skip bbox if too small.
                    bbox_width = xbr - xtl
                    bbox_height = ybr - ytl
                    if self.filter_bbox:
                        skip_bbox = check_filter_bboxes(
                            atis_yönelim_label, bbox_width=bbox_width, bbox_height=bbox_height
                        )
                        if skip_bbox:
                            continue
                    img_path = (
                        self.raw_all_dataset
                        / atis_yönelim_label
                        / f"cerkez_{scene_name}_{frame_index}_{str(total_bbox_count)}.jpg"
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
        self.process(xml_files_cerkez, video_files)


if __name__ == "__main__":
    dataset_folder_cerkez = Path("/home/utku/Documents/raw_datasets/dataset_new_cerkez")
    dataset_folder_dfas = Path("/home/utku/Documents/raw_datasets/dataset_new_dfas")
    dataset_folder_serefli3 = Path("/home/utku/Documents/raw_datasets/dataset_new_serefli3")

    parser_ser3 = ic(
        DataParserDFAS(
            dataset_folder=dataset_folder_serefli3,
            out_classification_dataset="atis_yönelim_classification_dataset/classification_dataset_ser3",
            filter_bbox=False,
            clear_prev_files=True,
        )
    )
    ic(parser_ser3.parse(multi_proc=True))
    ic(split_dataset(parser_ser3, train_ratio=3 / 4, scene_name="ser3"))

    parser_dfas = ic(DataParserDFAS(dataset_folder=dataset_folder_dfas, filter_bbox=False))
    ic(parser_dfas.parse())
    ic(split_dataset(parser_dfas, scene_name="dfas"))

    parser_cerkez = ic(DataParserCerkez(dataset_folder=dataset_folder_cerkez, filter_bbox=False))
    ic(parser_cerkez.parse())
    ic(split_dataset(parser_cerkez, scene_name="cerkez"))

    dataset_list = [
        "atis_yönelim_classification_dataset/classification_dataset_cerkez",
        "atis_yönelim_classification_dataset/classification_dataset_dfas",
        "atis_yönelim_classification_dataset/classification_dataset_ser3",
    ]
    ic(
        combine_datasets(
            dataset_list, target_dataset=Path("atis_yönelim_classification_dataset/classification_dataset_combined")
        )
    )
