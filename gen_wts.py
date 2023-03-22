import sys
import argparse
import os
import struct
import torch
import torchvision
import copy
from collections import OrderedDict
from pathlib import Path

# from utils.torch_utils import select_device


NUMBER_OF_CLASSES = len({'Tank-M48': 0, 'Tank-M60': 1, 'Tank-leopard': 2})
PRINT_BOOL = False
EFFNET_MODEL_TYPE = "B0"
SOURCE_EFFNET_WTS_NAMES_PATH = "/home/utku/Documents/repos/dfas_classifier/trained_models/exp_light_dataset_v1_other_decimated_classifier_effnet_luke_b0/light_dataset_v1_other_decimated_classifier_effnet_f1.wts"
TARGET_EFFNET_WTS_NAMES_PATH = "effnet_wts_names.txt"
WRONG_EFFNET_WTS_PATH = Path("/home/utku/Documents/repos/dfas_classifier/trained_models/exp_light_dataset_v1_other_decimated_classifier_effnet_1_layer_freeze/light_dataset_v1_other_decimated_classifier_effnet_f1.wts")
CORRECTED_EFFNET_WTS_PATH = WRONG_EFFNET_WTS_PATH.with_stem(WRONG_EFFNET_WTS_PATH.stem + "_corrected")


class EffNetModel(torch.nn.Module):
    def __init__(self, EffNet, luke):
        super(EffNetModel, self).__init__()
        if luke:
            out_features = EffNet._fc.in_features
        else:
            out_features = EffNet.classifier[1].in_features
        self.feature_extractor = copy.deepcopy(EffNet)
        number_of_input_features = out_features
        hier_classification_head = torch.nn.Linear(in_features=number_of_input_features, out_features=NUMBER_OF_CLASSES)
        if luke:
            self.feature_extractor._fc = hier_classification_head
        else:
            self.feature_extractor.classifier[1] = hier_classification_head

    def forward(self, input_frame):
        output = self.feature_extractor(input_frame)
        return output

def get_effnet(model_type=EFFNET_MODEL_TYPE, luke=False):
    weights = None
    if model_type == "B1":
        effnet = torchvision.models.efficientnet_b1(weights=weights)
    elif model_type == "B0":
        if luke:
            from efficientnet_pytorch import EfficientNet
            effnet = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            effnet = torchvision.models.efficientnet_b0(weights=weights)
    return effnet

class ResNetModel(torch.nn.Module):
    def __init__(self, ResNet):
        super(ResNetModel, self).__init__()
        out_features = ResNet.fc.in_features
        self.feature_extractor = copy.deepcopy(ResNet)
        number_of_input_features = out_features  # 512 for resnet18
        hier_classification_head = torch.nn.Linear(number_of_input_features, NUMBER_OF_CLASSES)
        self.feature_extractor.fc = hier_classification_head

    def forward(self, input_frame):
        output = self.feature_extractor(input_frame)
        return output
    
def get_resnet():
    weights = None
    ResNet = torchvision.models.resnet18(weights=weights)
    return ResNet

def load_state_dict(model, model_state_dict, luke):
    try:
        # without dataparalel
        # make sure NUMBER_OF_CLASSES is set correctly.
        model.load_state_dict(model_state_dict)
    except:
        # with dataparalel
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # remove `module.` (change if necessary)
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    return model

def write_to_wts_file(wts_file, model):
    with open(wts_file, "w") as f:
        f.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {} ".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")

def generate_wts_names(luke_wts_file_path=SOURCE_EFFNET_WTS_NAMES_PATH, target_wts_names_path=TARGET_EFFNET_WTS_NAMES_PATH):
    with open(luke_wts_file_path, 'r') as source_file:
        with open(target_wts_names_path, "w") as target_file:
            for line in source_file:
                words = line.split()  # Splits the line into words using spaces as the delimiter
                if words:  # Check if the list is not empty (i.e., the line is not empty)
                    first_word = words[0]
                    target_file.write(first_word+"\n")

def convert_wts(wrong_effnet_wts_path=WRONG_EFFNET_WTS_PATH, luke_names_path=TARGET_EFFNET_WTS_NAMES_PATH, corrected_effnet_wts_path=CORRECTED_EFFNET_WTS_PATH):
    with open(wrong_effnet_wts_path, 'r') as source_file:
        with open(luke_names_path, "r") as luke_file:
            with open(corrected_effnet_wts_path, "w") as corrected_target_file:
                for wrong_line, correct_name in zip(source_file, luke_file):
                    correct_name = correct_name.replace("\n","")
                    words = wrong_line.split()  # Splits the line into words using spaces as the delimiter
                    if words:  # Check if the list is not empty (i.e., the line is not empty)
                        words[0] = correct_name
                        correct_line = " ".join(words)
                        corrected_target_file.write(correct_line+"\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert .pt file to .wts")
    parser.add_argument("-w", "--weights", help="Input weights (.pt) file path (required)")
    parser.add_argument("-o", "--output", help="Output (.wts) file path (optional)")
    parser.add_argument("-g", "--wts_name_generation", action="store_true", help="To generate the wts file with correct layer names for trt.")
    parser.add_argument("-c", "--wts_conversion", action="store_true", help="Convert regular wts file to luke wts format for trt.")
    parser.add_argument("-e", "--effnet", action="store_true", help="Default is resnet.")
    parser.add_argument("-l", "--luke", action="store_true", help="Default is not luke.")
    args = parser.parse_args()
    if args.weights:
        if not os.path.isfile(args.weights) :
            raise SystemExit("Invalid input file")
        if not args.output:
            args.output = os.path.splitext(args.weights)[0] + ".wts"
        elif os.path.isdir(args.output):
            args.output = os.path.join(args.output, os.path.splitext(os.path.basename(args.weights))[0] + ".wts")
    return args.weights, args.output, args.wts_name_generation, args.wts_conversion, args.effnet, args.luke


def main():
    pt_file, wts_file, wts_name_generation, wts_conversion, effnet, luke = parse_args()
    print("output:", wts_file)
    print("pt_file:", pt_file)
    print("generate_wts_names:", generate_wts_names)
    print("effnet:", effnet)
    print("luke:", luke)
    # Initialize
    device = "cpu"
    print(pt_file)
    # Load model
    if wts_name_generation:
        generate_wts_names()
    elif wts_conversion:
        convert_wts()
    elif pt_file:
        if effnet:
            effnet = get_effnet(luke=luke)
            model = EffNetModel(effnet, luke=luke)
        else:
            resnet = get_resnet()
            model = ResNetModel(resnet)
        # original regular saved file with or without DataParallel
        model_state_dict = torch.load(pt_file)
        model = load_state_dict(model=model, model_state_dict=model_state_dict, luke=luke)
        print(type(model))
        # model = model['ema' if model.get('ema') else 'model'].float()
        if PRINT_BOOL:
            print(model)
        else:
            model.to(device).eval()
            write_to_wts_file(wts_file=wts_file, model=model)


if __name__ == "__main__":
    main()
