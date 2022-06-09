from genericpath import exists
from lzma import is_check_supported
import random
from tabnanny import check
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from pathlib import Path
from networkx.algorithms.dag import ancestors
import networkx as nx
import shutil
import subprocess 
from matplotlib import pyplot as plt
# from scripts.hier_utils import c_dict

"""
Create classification dataset for torchvision. Folders should be named after classes in train/val/test folders.
"""

# TODO LIST
# Take image folder and split labels and images to combined dataset from different xml files and image folders of sepearate tasks.
#

# Dfas needs the graph below
dfas_tree = nx.DiGraph()
dfas_tree.add_node('araç')
dfas_tree.add_node('insan')
dfas_tree.add_edge('insan', 'askeri insan')
dfas_tree.add_edge('askeri', 'silahlı asker')
dfas_tree.add_edge('askeri', 'silahsız asker')
dfas_tree.add_edge('insan', 'sivil insan')
dfas_tree.add_edge('araç', 'sivil araç')
dfas_tree.add_edge('araç', 'askeri araç')
dfas_tree.add_edge('askeri araç', 'lastikli')
dfas_tree.add_edge('lastikli', 'Top-lastikli')
dfas_tree.add_edge('lastikli', 'lastikli araç')
dfas_tree.add_edge('askeri araç', 'paletli')
dfas_tree.add_edge('paletli', 'Tank')
dfas_tree.add_edge('paletli', 'ZPT')
dfas_tree.add_edge('paletli', 'Top-paletli')
dfas_tree.add_edge('Tank', 'Tank-leopard')
dfas_tree.add_edge('Tank', 'Tank-M60')
dfas_tree.add_edge('Tank', 'Tank-M48')
dfas_tree.add_edge('Tank', 'Tank-fırtına')

# Cerkezköy needs the graph below
cerkez_tree = nx.DiGraph()
cerkez_tree.add_node('insan')
cerkez_tree.add_node('arac')
cerkez_tree.add_edge('insan', 'askeri')
cerkez_tree.add_edge('askeri', 'silahlı')
cerkez_tree.add_edge('askeri', 'silahsız')
cerkez_tree.add_edge('insan', 'sivil')
cerkez_tree.add_edge('arac', 'sivil arac')
cerkez_tree.add_edge('arac', 'askeri arac')
cerkez_tree.add_edge('askeri arac', 'lastikli')
cerkez_tree.add_edge('askeri arac', 'paletli')
cerkez_tree.add_edge('paletli', 'tank')
cerkez_tree.add_edge('paletli', 'ZMA')
cerkez_tree.add_edge('tank', 'leopard')
cerkez_tree.add_edge('tank', 'm60')
cerkez_tree.add_edge('tank', 'm48')

# c_dict = {'insan':0, 'arac':1, 'askeri':2, 'sivil':3, 'silahlı':4, 'silahsız':5, \
#     'sivil arac':6, 'askeri arac':7, 'lastikli':8, 'paletli':9, 'tank':10, 'ZMA':11,\
#     'leopard':12, 'm60':13, 'm48':14}


# TODO discarded "atış", "araçlı_atış", "siper_mevzi" for now
c_dict = {'araç': 0, 'insan': 1, 'askeri araç': 2, 'sivil araç': 3, 'askeri insan': 4, 'sivil insan': 5,
          'lastikli': 6, 'paletli': 7, 'silahlı asker': 8, 'silahsız asker': 9, 'Top-lastikli': 10, 'lastikli araç': 11,
          'Tank': 12, 'Top-paletli': 13, 'ZPT': 14, 'Tank-M48': 15, 'Tank-M60': 16, 'Tank-fırtına': 17, 'Tank-leopard': 18}

dfas_classes = list(c_dict.keys())

dfas_to_cerkez_lookup = {'araç': 'arac', 'askeri araç': 'askeri arac', 'sivil araç': 'sivil arac', 'askeri insan': 'askeri', 'sivil insan': 'sivil',
                         'silahlı asker': 'silahlı', 'silahsız asker': 'silahsız', 'lastikli araç': 'lastikli',
                         'Tank': 'tank', 'ZPT': 'ZMA', 'Tank-M48': 'm48', 'Tank-M60': 'm60', 'Tank-leopard': 'leopard'}
cerkez_to_dfas_lookup = {value: key for key, value in dfas_to_cerkez_lookup.items()}


class DataParserDFAS():
    def __init__(self, dataset_folder: Path, filter_bbox:bool = False):
        self.filter_bbox = filter_bbox
        self.dataset_folder = dataset_folder
        self.yolo_dataset = Path("hier_clasification_dataset/clasification_dataset_dfas")
        self.raw_all_dataset = self.yolo_dataset / "all" 
        self.create_directories()

        # Get xml files and image folders.
        self.dataset_folder_paths = list(self.dataset_folder.iterdir())
        self.xml_files = self.get_sorted_xml_files_paths(self.dataset_folder_paths)
        assert len(self.xml_files) == len(self.dataset_folder_paths) / \
            2, "# of xml files is not equal to # of image folders in the path."
        self.image_folders = self.get_sorted_image_folder_paths(
            self.dataset_folder_paths, self.xml_files)
        assert len(self.xml_files) == len(
            self.image_folders), "# of xml files is not equal to # of image folders after choosing only available label folders."
        self.label_and_image_paths = list(zip(self.xml_files, self.image_folders))
        print(f"There are {len(self.label_and_image_paths)} labelled image folders available.")

    def create_directories(self):
        # CREATE directories
        for class_name in dfas_classes:
            (self.raw_all_dataset / class_name).mkdir(exist_ok=True, parents=True)
        for split_type in ["train", "val", "test"]:
            (self.yolo_dataset / split_type).mkdir(exist_ok=True)
            for class_name in dfas_classes:
                (self.yolo_dataset / split_type / class_name).mkdir(exist_ok=True)

    def get_sorted_xml_files_paths(self, dataset_folder_paths: list):
        xml_files = []
        for path in dataset_folder_paths:
            if path.suffix == ".xml":
                xml_files.append(path)
        xml_files.sort()
        return xml_files

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

    def process(self, image_props: list):
        total_bbox_count = 0
        for xmlfile, image_folder in self.label_and_image_paths:
            image_height = image_props[xmlfile]["height"]
            image_width = image_props[xmlfile]["width"]
            image_path = image_props[xmlfile]["path"]
            print(xmlfile)
            print()
            xml_tree = ET.parse(xmlfile)
            root = xml_tree.getroot()
            all_images = root.findall("image")
            for img in all_images:
                # img = next(images)
                img_name = img.get("name")
                image_path = image_folder / img_name
                cv2_image = cv2.imread(str(image_path))
                for box in img.iter('box'):
                    # label extraction.
                    # includes all classes (without doğrultmuş/doğrultmamış ve lastik_palet_izi.)
                    label = box.get('label')
                    # TODO siper_mevzi missing
                    if label in ["atış", "araçlı_atış", "siper_mevzi"]:
                        continue
                    # Fix the label for some wrongly named labels in annotations.
                    if label in ["silahlı insan", "silahsız insan"]:
                        label = label.replace("insan", "asker")
                    box_dict = {}
                    box_dict['label'] = label
                    # NOTE we do not care about attirbutes (doğrultmuş/doğrultmamış) yet.
                    xtl = int(round(float(box.get('xtl'))))
                    if xtl < 0:
                        xtl = 0
                    ytl = int(round(float(box.get('ytl'))))
                    xbr = int(round(float(box.get('xbr'))))
                    ybr = int(round(float(box.get('ybr'))))

                    img_size = image_width * image_height
                    if img_size < 1200:
                        continue  # Discard if image resolution is less than 1200.
                    # for output vector
                    # YOLO label conversion
                    # Normalize the pixel coordinates for yolo
                    bbox_width = xbr - xtl
                    bbox_height = ybr - ytl

                    # Skip bbox if too small.
                    if self.filter_bbox:
                        skip_bbox = check_filter_bboxes(label, bbox_width=bbox_width, bbox_height=bbox_height)
                        if skip_bbox:
                            continue
                    if label =="lastikli askeri araç":
                        label = 'lastikli araç'
                    img_path = self.raw_all_dataset / label / f'dfas_{str(total_bbox_count)}.jpg'
                    cropped_image = cv2_image[ytl:ybr, xtl:xbr, :]
                    cv2.imwrite(str(img_path), cropped_image)
                    total_bbox_count += 1

    def parse(self):
        img_dims_dict = {}
        for xml_file, image_folder_path in self.label_and_image_paths:
            image_folder_list = list(im for im in image_folder_path.iterdir()
                                     if im.suffix in [".png", ".jpg", ".jpeg"])
            an_image_path_in_folder = image_folder_list[0]
            print(an_image_path_in_folder)
            temp_img = cv2.imread(str(an_image_path_in_folder))
            height, width, channels = temp_img.shape
            img_dims_dict[xml_file] = {"width": width, "height": height, "path": image_folder_path}
        self.process(img_dims_dict)


    def split_dataset(self, train_ratio = 4/5):
        """Hardcoded dataset split."""
        def plot_bar_graph_per_split(classes_per_split:dict, plot_name:str = "train"):
            classes = list(classes_per_split.keys())
            values = list(classes_per_split.values())
            plt.figure(figsize =(10, 7))
            plt.barh(classes, values)
            plt.xlabel("Classes")
            plt.ylabel("No of elements")
            plt.title(plot_name)
            plt.savefig(plot_name + "_dfas.jpg")
            plt.clf()
        # 0.75, 0.125, 0.125 -> train,val,test split
        cmd = f"find {self.raw_all_dataset} -type f | wc -l"
        num_of_files_stdout = subprocess.run(cmd, check = True, shell = True, stdout=subprocess.PIPE)
        number_of_images = int(str(num_of_files_stdout.stdout).replace("\\n", "").replace("b", "").replace("'", ""))
        print("Number of images for classification in dfas dataset folder: ", number_of_images)
        train_end_index = int(number_of_images * train_ratio)
        print("Number of images training images: ", train_end_index)
        number_per_classes_per_split = {data_split: {class_name: 0 for class_name in dfas_classes} for data_split in  ["train","val","test"]}
        for class_folder in self.raw_all_dataset.iterdir():
            for image_in_class in class_folder.iterdir():
                criterion = random.randint(0, 1)
                if image_in_class.is_file():
                    image_index = int(image_in_class.stem.replace("dfas_",""))
                    if image_index <= train_end_index:
                        data_split = "train"
                    else:
                        if criterion == 0:
                            data_split = "val"
                        elif criterion == 1:
                            data_split = "test"
                    number_per_classes_per_split[data_split][class_folder.name] += 1
                    target_image_path = (self.yolo_dataset / data_split)/ class_folder.name / image_in_class.name
                    shutil.copyfile(image_in_class, target_image_path)
        print(number_per_classes_per_split)
        for split_name, classes_per_split in number_per_classes_per_split.items():
            plot_bar_graph_per_split(classes_per_split, plot_name=split_name)


class DataParserCerkez():
    def __init__(self, dataset_folder=Path, filter_bbox:bool = False):
        self.filter_bbox = filter_bbox
        self.dataset_folder = dataset_folder
        self.yolo_dataset = Path("hier_clasification_dataset/clasification_dataset_cerkez")
        self.raw_all_dataset = self.yolo_dataset / "all" 
        self.create_directories()

    def create_directories(self):
        # CREATE directories
        for class_name in dfas_classes:
            (self.raw_all_dataset / class_name).mkdir(exist_ok=True, parents=True)
        # Next level dataset split folders.
        for split_type in ["train", "val", "test"]:
            (self.yolo_dataset / split_type).mkdir(exist_ok=True)
            for class_name in dfas_classes:
                (self.yolo_dataset / split_type / class_name).mkdir(exist_ok=True)

    def process(self, xmlfiles: list, videofiles: list) -> None:
        """Process previous cerkez labels to match new dfas label style."""
        total_bbox_count = 0
        length = len(xmlfiles)
        for k in range(length):
            video = videofiles[k]
            xmlfile = xmlfiles[k]
            cap = cv2.VideoCapture(str(video))
            xml_tree = ET.parse(xmlfile)
            root = xml_tree.getroot()
            images = root.iter('image')
            while cap.isOpened():
                retval, frame = cap.read()
                if not retval:
                    break
                img = next(images)
                for box in img.iter('box'):
                    label_list = []
                    # label extraction.
                    label = box.get('label')  # arac OR insan
                    box_dict = {}
                    box_dict['label'] = label
                    for attr in box.iter('attribute'):
                        attr_name = attr.get('name')
                        answer = attr.text
                        box_dict[attr_name] = answer
                    print(box_dict)
                    if label == "insan":
                        label_list.append(label)
                        label_list.append(box_dict['tür'])
                        label_list.append(box_dict['silah nitelik'])
                    else:
                        if box_dict['ek nitelik'] != "yok":
                            supers = list(ancestors(cerkez_tree, box_dict['ek nitelik']))
                            label_list = label_list + supers
                            label_list.append(box_dict['ek nitelik'])
                        else:
                            label_list.append(label)
                            label_list.append(box_dict['tür'] + " arac")
                    most_specific_label = label_list[-1]
                    print(most_specific_label)

                    # top left and bottom right label_list are available.
                    xtl = int(round(float(box.get('xtl'))))
                    if xtl < 0:
                        xtl = 0
                    ytl = int(round(float(box.get('ytl'))))
                    xbr = int(round(float(box.get('xbr'))))
                    ybr = int(round(float(box.get('ybr'))))

                    # Skip bbox if too small.
                    bbox_width = xbr - xtl
                    bbox_height = ybr - ytl
                    if self.filter_bbox:
                        skip_bbox = check_filter_bboxes(label, bbox_width=bbox_width, bbox_height=bbox_height)
                        if skip_bbox:
                            continue

                    # dfas_version_of_label is different than label if available in lookup.
                    if "Cerkezkoy-1_Speedup_6" in str(xmlfile):
                        if most_specific_label == "m60":
                            most_specific_label = 'Tank-fırtına'
                    dfas_version_of_label = most_specific_label if (cerkez_to_dfas_lookup.get(
                        most_specific_label) is None) else cerkez_to_dfas_lookup[most_specific_label]

                    img_path = self.raw_all_dataset / dfas_version_of_label / f'cerkez_{str(total_bbox_count)}.jpg'
                    cropped_image = frame[ytl:ybr, xtl:xbr, :]
                    cv2.imwrite(str(img_path), cropped_image)
                    total_bbox_count += 1
                # Write image specific labels to yolo label txt

    def split_dataset(self, train_ratio = 4/5):
        """Hardcoded dataset split."""
        def plot_bar_graph_per_split(classes_per_split:dict, plot_name:str = "train"):
            classes = list(classes_per_split.keys())
            values = list(classes_per_split.values())
            plt.figure(figsize =(10, 7))
            plt.barh(classes, values)
            plt.xlabel("Classes")
            plt.ylabel("No of elements")
            plt.title(plot_name)
            plt.savefig(plot_name + "_cerkez.jpg")
            plt.clf()
        # 0.75, 0.125, 0.125 -> train,val,test split
        cmd = f"find {self.raw_all_dataset} -type f | wc -l"
        num_of_files_stdout = subprocess.run(cmd, check = True, shell = True, stdout=subprocess.PIPE)
        number_of_images = int(str(num_of_files_stdout.stdout).replace("\\n", "").replace("b", "").replace("'", ""))
        print("Number of images for classification in cerkez dataset folder: ", number_of_images)
        train_end_index = int(number_of_images * train_ratio)
        print("Number of images training images: ", train_end_index)
        number_per_classes_per_split = {data_split: {class_name: 0 for class_name in dfas_classes} for data_split in  ["train","val","test"]}
        for class_folder in self.raw_all_dataset.iterdir():
            for image_in_class in class_folder.iterdir():
                criterion = random.randint(0, 1)
                if image_in_class.is_file():
                    image_index = int(image_in_class.stem.replace("cerkez_",""))
                    if image_index <= train_end_index:
                        data_split = "train"
                    else:
                        if criterion == 0:
                            data_split = "val"
                        elif criterion == 1:
                            data_split = "test"
                    number_per_classes_per_split[data_split][class_folder.name] += 1
                    target_image_path = (self.yolo_dataset / data_split)/ class_folder.name / image_in_class.name
                    shutil.copyfile(image_in_class, target_image_path)
        print(number_per_classes_per_split)
        for split_name, classes_per_split in number_per_classes_per_split.items():
            plot_bar_graph_per_split(classes_per_split, plot_name=split_name)

    def get_sorted_xml_files_paths(self, dataset_folder_paths: list):
        xml_files = []
        for path in dataset_folder_paths:
            if path.suffix == ".xml":
                xml_files.append(path)
        xml_files.sort()
        return xml_files

    def parse(self):
        dataset_folder_paths_cerkez = list(self.dataset_folder.iterdir())
        xml_files_cerkez = self.get_sorted_xml_files_paths(dataset_folder_paths_cerkez)
        assert len(xml_files_cerkez) == len(dataset_folder_paths_cerkez) / \
            2, "# of xml files is not equal to # of image folders in the path."
        video_files = []
        for folder in dataset_folder_paths_cerkez:
            if folder.is_dir():
                video_path = (folder / folder.stem).with_suffix(".mp4")
                video_files.append(video_path)
        video_files.sort()
        self.process(xml_files_cerkez, video_files)


# Skip the label if the width and height are not appropriate
def check_filter_bboxes(label, bbox_width, bbox_height ,car_h_thr = 26, car_wh_thr = 18, person_h_thr = 20, person_wh_thr = 10):
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
    target_dataset.mkdir(exist_ok = True)
    for dataset in dataset_list:
        # Copy with update, adds missing images to matching folders.
        cmd = f"cp -ur {dataset}/* {target_dataset}/"
        subprocess.run(cmd, check=True, shell=True)


if __name__ == "__main__":
    print("class_list: ", c_dict.keys())
    print("Number of classes: ", len(c_dict.keys()))
    dataset_folder_cerkez = Path("dataset_new_cerkez")
    dataset_folder_dfas = Path("dataset_new_dfas")
    
    parser_dfas=DataParserDFAS(dataset_folder=dataset_folder_dfas, filter_bbox=False)
    parser_dfas.parse()
    parser_dfas.split_dataset()

    parser_cerkez = DataParserCerkez(dataset_folder=dataset_folder_cerkez,filter_bbox=False)
    parser_cerkez.parse()
    parser_cerkez.split_dataset()

    dataset_list = ["hier_clasification_dataset/clasification_dataset_cerkez", "hier_clasification_dataset/clasification_dataset_dfas"]
    combine_datasets(dataset_list, target_dataset=Path("hier_clasification_dataset/classification_dataset_combined"))
