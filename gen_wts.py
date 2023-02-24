import sys
import argparse
import os
import struct
import torch
import torchvision
import copy
from collections import OrderedDict

# from utils.torch_utils import select_device


NUMBER_OF_CLASSES = len({'Tank-M48': 0, 'Tank-M60': 1, 'Tank-leopard': 2})
PRINT_BOOL = False
EFFNET_MODEL_TYPE = "B0"


class EffNetModel(torch.nn.Module):
    def __init__(self, EffNet):
        super(EffNetModel, self).__init__()
        out_features = EffNet.classifier[1].in_features
        self.feature_extractor = copy.deepcopy(EffNet)
        number_of_input_features = out_features
        hier_classification_head = torch.nn.Linear(in_features=number_of_input_features, out_features=NUMBER_OF_CLASSES)
        self.feature_extractor.classifier[1] = hier_classification_head

    def forward(self, input_frame):
        output = self.feature_extractor(input_frame)
        return output

def get_effnet(model_type=EFFNET_MODEL_TYPE):
    weights = None
    if model_type == "B1":
        effnet = torchvision.models.efficientnet_b1(weights=weights)
    elif model_type == "B0":
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

def load_state_dict(model, model_state_dict):
    try:
        # without dataparalel
        # make sure NUMBER_OF_CLASSES is set correctly.
        model.load_state_dict(model_state_dict)
    except:
        # with dataparalel
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = "feature_extractor." + k[7:]  # remove `module.`
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

def parse_args():
    parser = argparse.ArgumentParser(description="Convert .pt file to .wts")
    parser.add_argument("-w", "--weights", required=True, help="Input weights (.pt) file path (required)")
    parser.add_argument("-o", "--output", help="Output (.wts) file path (optional)")
    parser.add_argument("-r", "--effnet", action="store_true", help="Default is resnet.")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid input file")
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + ".wts"
    elif os.path.isdir(args.output):
        args.output = os.path.join(args.output, os.path.splitext(os.path.basename(args.weights))[0] + ".wts")
    return args.weights, args.output, args.effnet


def main():
    pt_file, wts_file, effnet = parse_args()
    print("output:", wts_file)
    print("pt_file:", pt_file)
    # Initialize
    device = "cpu"
    print(pt_file)
    # Load model
    if effnet:
        effnet = get_effnet()
        model = EffNetModel(effnet)
    else:
        resnet = get_resnet()
        model = ResNetModel(resnet)
    # original regular saved file with or without DataParallel
    model_state_dict = torch.load(pt_file)
    model = load_state_dict(model=model, model_state_dict=model_state_dict)
    print(type(model))
    # model = model['ema' if model.get('ema') else 'model'].float()
    if PRINT_BOOL:
        print(model)
    else:
        model.to(device).eval()
        write_to_wts_file(wts_file=wts_file, model=model)


if __name__ == "__main__":
    main()
