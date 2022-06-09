import torch
import torchvision
import copy
from base_dataloader import class_names, hyps

# model = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
# self.Resnet18 = torchvision.models.resnet50()
# NOTE should I feed Conv or regular NN output to LSTM. What is the difference?
# Might be useful : https://arxiv.org/pdf/2105.03186.pdf
NUM_DISTORTION = 18
NUM_SEVERITY = 4
NUM_OF_OUTPUT_CLASSES = len(class_names)
GPU_ID0 = hyps["GPUS"][0]
GPUS = hyps["GPUS"]

ResNet50 = True
if ResNet50:
    ResNet = torchvision.models.resnet50(pretrained=True)
else:
    ResNet = torchvision.models.resnet18(pretrained=True)

# Freeze first 2 layers
if hyps["FREEZE"]:
    for idx, (name, param) in enumerate(ResNet.named_parameters()):
        if idx <= 2 or "layer1" in name or "layer2" in name:
            param.requires_grad = False


out_features = ResNet.fc.in_features
# Resnet18.fc = torch.nn.Identity()
# Resnet18_feature_extractor = Resnet18
# Naive Model for Barış's code
class BaseModel(torch.nn.Module):
    def __init__(self, feature_extractor = ResNet):
        super(BaseModel,self).__init__()
        # number_of_input_features is 2048 (Resnet50.fc.in_features)
        self.feature_extractor = copy.deepcopy(feature_extractor)
        self.feature_extractor.fc = torch.nn.Identity()
        number_of_input_features = out_features #512 for resnet18
        self.fc_naive = torch.nn.Linear(number_of_input_features, NUM_OF_OUTPUT_CLASSES)
        # Feed LSTM features to attention.
        # self.LSTM = torch.

    def forward(self, input_frame):
        x = self.feature_extractor(input_frame)
        output = self.fc_naive(x)
        return output

base_net_single = BaseModel()
print("Currently selected GPUS: ",GPUS)
base_net = torch.nn.DataParallel(base_net_single, device_ids = GPUS)