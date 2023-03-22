import torch
import torchvision
from matplotlib import pyplot as plt
import copy
from PIL import Image
from dataloader import DEVICE, GPU_IDS, C_DICT, HYPS
from typing import OrderedDict

# TODO maybe freeze some layers.
LUKE = HYPS["LUKE"]
EFFNET_WEIGHTS_DICT = {"B0": 224, "B1": 240}
MODEL = HYPS["MODEL"]
OTHER_HEAD = HYPS["OTHER_HEAD"]
FROZEN_LAYERS = HYPS["FROZEN_LAYERS"]
FREEZE = HYPS["FREEZE"]
PRETRAINED = HYPS["PRETRAINED"]
MODEL_TYPE = HYPS["model_type"]
if OTHER_HEAD:
    NUMBER_OF_CLASSES = len(C_DICT) + 1
else:
    NUMBER_OF_CLASSES = len(C_DICT)


def freeze_parameter(param, freeze_param_bool):
    param.requires_grad = not freeze_param_bool

def get_frozen_layer_names(prefix):
    frozen_layer_names = []
    for frozen_layer_idx in FROZEN_LAYERS:
        frozen_layer_names.append(f"{prefix}{frozen_layer_idx}")
    return frozen_layer_names

def freeze_model_layers(network, initial_layers_count, prefix):
    frozen_layer_names = get_frozen_layer_names(prefix=prefix)
    for idx, (name, param) in enumerate(network.named_parameters()):
        freeze_layer = False
        for layer_name in frozen_layer_names:
            if idx < initial_layers_count or layer_name in name:
                freeze_layer = True
        freeze_parameter(param, freeze_layer)


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

def get_effnet(luke):
    weights = None
    if MODEL_TYPE == "B1":
        if PRETRAINED:
            weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
        effnet = torchvision.models.efficientnet_b1(weights=weights)
    elif MODEL_TYPE == "B0":
        if luke:
            from efficientnet_pytorch import EfficientNet
            effnet = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            if PRETRAINED:
                weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            effnet = torchvision.models.efficientnet_b0(weights=weights)
    elif MODEL_TYPE == "s":
        if PRETRAINED:
            weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        effnet = torchvision.models.efficientnet_v2_s(weights=weights)
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
    if PRETRAINED:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ResNet = torchvision.models.resnet18(weights=weights)
    return ResNet

# resnet18
if MODEL == "resnet18":
    ResNet = get_resnet()
    # Freeze first 2 layers
    if FREEZE:
        resnet_initial_layers_count = 3
        freeze_model_layers(ResNet, resnet_initial_layers_count, prefix="layers")
    net = ResNetModel(ResNet)
# effnet b0
elif MODEL == "effnet":
    prefix = LUKE*"_blocks." + (not LUKE)*"features." # TODO check if this is correct.
    EffNet = get_effnet(luke=LUKE)
    if FREEZE:
        effnet_initial_layers_count = 3
        freeze_model_layers(EffNet, effnet_initial_layers_count, prefix=prefix)
    net = EffNetModel(EffNet, luke=LUKE)

single_net = net.to(DEVICE)
net = single_net
print(GPU_IDS)
# Multi GPU train
net = torch.nn.DataParallel(single_net, device_ids=GPU_IDS)
# NOTE you can add atış yonelimi with the head below.
# direction_of_fire_classification_head = torch.nn.Linear(in_features = net.fc.in_features, out_features = NUMBER_OF_CLASSES)

if __name__ == "__main__":
    print(net)
