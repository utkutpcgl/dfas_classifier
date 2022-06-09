import torch
from matplotlib import pyplot as plt
import copy
from PIL import Image
from dataloader import DEVICE, GPU_IDS, C_DICT, DFAS_TREE


NUMBER_OF_CLASSES = len(C_DICT)
single_net = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0", pretrained=True)
hier_classification_head = torch.nn.Linear(
    in_features=single_net.classifier.fc.in_features, out_features=NUMBER_OF_CLASSES
)
single_net.classifier.fc = hier_classification_head

# Multi GPU train
net = single_net
# net = torch.nn.DataParallel(net, device_ids = GPU_IDS)
# NOTE you can add atış yönelimi with the head below.
# direction_of_fire_classification_head = torch.nn.Linear(in_features = net.fc.in_features, out_features = NUMBER_OF_CLASSES)


def main():
    pass


if __name__ == "__main__":
    # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
    # net.eval().to("cuda")
    # uris = [
    #     'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    #     'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    #     'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    #     'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
    # ]
    # batch = torch.cat([utils.prepare_input_from_uri(uri) for uri in uris]).to("cuda")
    # with torch.no_grad():
    #     output = torch.nn.functional.softmax(net(batch), dim = 1)
    # results = utils.pick_n_best(predictions = output, n = 5)
    # for uri, result in zip(uris, results):
    #     img = Image.open(requests.get(uri, stream=True).raw)
    #     img.thumbnail((256,256), Image.ANTIALIAS)
    #     plt.imshow(img)
    #     plt.show()
    #     print(result)
    print(net)
