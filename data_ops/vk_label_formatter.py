"""Create an ordered dataset directory that can be fed to yolo dataset creator scripts."""

from argparse import ArgumentParser
from pathlib import Path
from subprocess import run

CERKEZ_PATH = Path("dataset_new_cerkez")
DFAS_PATH = Path("dataset_new_dfas")

CERKEZ_ZIP_FILES = [
    zip_file for zip_file in CERKEZ_PATH.iterdir() if zip_file.is_file() if "label_folder" not in zip_file.name
]
DFAS_ZIP_FILES = [
    zip_file for zip_file in DFAS_PATH.iterdir() if zip_file.is_file() if "label_folder" not in zip_file.name
]


def rename_files_cerkez():
    if len(CERKEZ_ZIP_FILES) != 0:
        for zip_file in CERKEZ_ZIP_FILES:
            # Add _label_folder to the end before the suffix.
            new_zip_file = zip_file.with_name(zip_file.stem + "_label_folder.zip")
            # Remove "task_" and "-cvat for images 1.1"
            new_zip_name = new_zip_file.name.replace("task_", "").replace("-cvat for images 1.1", "")
            # cerkez -> Cerkez, speed -> Speed
            new_zip_name = new_zip_name.replace("cerkez", "Cerkez").replace("speed", "Speed")
            # Rename the file with the path protected.
            new_zip_file = zip_file.with_name(new_zip_name)
            if zip_file.name != new_zip_name:
                zip_file.rename(zip_file.with_name(new_zip_name))


def rename_files_dfas():
    if len(DFAS_ZIP_FILES) != 0:
        for zip_file in DFAS_ZIP_FILES:
            # Add _label_folder to the end before the suffix.
            new_zip_file = zip_file.with_name(zip_file.stem + "_label_folder.zip")
            # Remove "task_" and "-cvat for images 1.1"
            new_zip_name = new_zip_file.name.replace("task_", "").replace("-cvat for images 1.1", "")
            # Rename the file with the path protected.
            new_zip_file = zip_file.with_name(new_zip_name)
            if zip_file.name != new_zip_name:
                zip_file.rename(zip_file.with_name(new_zip_name))


def unzip_folders(dataset_path: Path) -> list:
    cerkez_corrected_zip_files = [zip_file for zip_file in dataset_path.iterdir() if zip_file.suffix == ".zip"]
    print(cerkez_corrected_zip_files)
    cerkez_zip_to_proccessed = [
        zip_file for zip_file in cerkez_corrected_zip_files if not zip_file.with_name(zip_file.stem).exists()
    ]
    print(cerkez_zip_to_proccessed)
    # Unzip all zip files.
    for zip_file in cerkez_zip_to_proccessed:
        cmd = f"unzip {zip_file} -d {zip_file.with_name(zip_file.stem)}"
        run(cmd, shell=True, check=True)
    label_folders = [folder.with_name(folder.stem) for folder in cerkez_corrected_zip_files]
    return label_folders


def mv_xml_files(label_folders: list):
    for label_folder in label_folders:
        target_file_path = (label_folder.parent / label_folder.name.replace("_label_folder", "")).with_suffix(".xml")
        if not target_file_path.exists():
            cmd = f"mv {label_folder}/annotations.xml {target_file_path}"
            run(cmd, shell=True, check=True)


def rm_artifact_folders(label_folders: list):
    for label_folder in label_folders:
        label_folder.rmdir()


def main():
    parser = ArgumentParser()
    parser.add_argument("--cerkez", action="store_true", help="format dfas dataset")
    parser.add_argument("--dfas", action="store_true", help="format dfas dataset")
    opt = parser.parse_args()
    cerkez = opt.cerkez
    dfas = opt.dfas
    if cerkez:
        rename_files_cerkez()
        label_folders = unzip_folders(CERKEZ_PATH)
        mv_xml_files(label_folders=label_folders)
        # rm_artifact_folders(label_folders)
    if dfas:
        rename_files_dfas()
        label_folders = unzip_folders(DFAS_PATH)
        mv_xml_files(label_folders=label_folders)
        rm_artifact_folders(label_folders)


if __name__ == "__main__":
    main()
