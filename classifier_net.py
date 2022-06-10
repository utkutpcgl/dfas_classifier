import torch
import torchvision
from matplotlib import pyplot as plt
import copy
from PIL import Image
from dataloader import DEVICE, GPU_IDS, C_DICT, hyps

# TODO maybe freeze some layers.

model = hyps["MODEL"]
NUMBER_OF_CLASSES = len(C_DICT)

# resnet18
if model == "resnet":
    ResNet = torchvision.models.resnet18(pretrained=True)
    # Freeze first 2 layers
    if hyps["FREEZE"]:
        for idx, (name, param) in enumerate(ResNet.named_parameters()):
            if idx <= 2 or "layer1" in name or "layer2" in name:
                param.requires_grad = False
    out_features = ResNet.fc.in_features

    class ResNetModel(torch.nn.Module):
        def __init__(self, feature_extractor=ResNet):
            super(ResNetModel, self).__init__()
            # number_of_input_features is 2048 (Resnet50.fc.in_features)
            self.feature_extractor = copy.deepcopy(feature_extractor)
            number_of_input_features = out_features  # 512 for resnet18
            self.feature_extractor.fc = torch.nn.Linear(number_of_input_features, NUMBER_OF_CLASSES)

        def forward(self, input_frame):
            output = self.feature_extractor(input_frame)
            return output

    net = ResNetModel()

# effnet b0
elif model == "effnet":
    effnet_single = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0", pretrained=True)
    if hyps["FREEZE"]:
        pass
    hier_classification_head = torch.nn.Linear(
        in_features=effnet_single.classifier.fc.in_features, out_features=NUMBER_OF_CLASSES
    )
    effnet_single.classifier.fc = hier_classification_head

    # Multi GPU train
    net = effnet_single

# net = torch.nn.DataParallel(net, device_ids = GPU_IDS)
# NOTE you can add atış yönelimi with the head below.
# direction_of_fire_classification_head = torch.nn.Linear(in_features = net.fc.in_features, out_features = NUMBER_OF_CLASSES)


def main():
    pass


if __name__ == "__main__":
    print(net)
