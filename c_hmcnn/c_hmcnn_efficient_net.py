import torch
import torchvision
from matplotlib import pyplot as plt
import copy
from PIL import Image
import networkx as nx
from dataloader import DEVICE, GPU_IDS, C_DICT, DFAS_TREE


NUMBER_OF_CLASSES = len(C_DICT)
single_eff_net = torchvision.models.regnet_x_16gf(pretrained=True)
hier_classification_head = torch.nn.Linear(in_features = single_eff_net.fc.in_features, out_features =  NUMBER_OF_CLASSES)
single_eff_net.fc = hier_classification_head

# single_eff_net = torchvision.models.efficientnet_b0(pretrained=True)# torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# # hier_classification_head = torch.nn.Linear(in_features = single_eff_net.classifier.fc.in_features, out_features =  NUMBER_OF_CLASSES)
# hier_classification_head = torch.nn.Linear(in_features = single_eff_net.classifier[1].in_features, out_features =  NUMBER_OF_CLASSES)
# single_eff_net.classifier[1] = hier_classification_head

# NOTE you can add atış yönelimi with the head below.
# direction_of_fire_classification_head = torch.nn.Linear(in_features = eff_net.fc.in_features, out_features = NUMBER_OF_CLASSES)

def get_constr_out(output_logits, R_constraint_matrix):
    """ Given the output of the neural network output_logits returns the output of MCM given the hierarchy constraint expressed in the matrix R 
    Returns maximum value (not index) (should return max of descendents -> that is the constraint)"""
    # TODO In evaulation only the max conf path should stay alive. Kill other sigmoid outputs.
    # (not possible combinations) should be eliminated.
    c_out = output_logits.double()
    c_out = c_out.unsqueeze(1)
    # len(output_logits) becomes batch size during training.
    c_out = c_out.expand(len(output_logits), R_constraint_matrix.shape[1], R_constraint_matrix.shape[1])
    R_batch = R_constraint_matrix.expand(len(output_logits), R_constraint_matrix.shape[1], R_constraint_matrix.shape[1])
    constr_r = c_out.clone()
    constr_r[R_batch == 0] = -float('inf') # This is like R_batch*c_out
    final_out, _ = torch.max(constr_r, dim = 2)
    return final_out

class ConstrainedEffNet(torch.nn.Module):
    def __init__(self, R_constraint_matrix, eff_net):
        super(ConstrainedEffNet,self).__init__()
        self.constrained_effnet = copy.deepcopy(eff_net)
        self.R_constraint_matrix = R_constraint_matrix
    
    def forward(self, x):
        x = self.constrained_effnet(x)
        # NOTE training is a built-in variable of torch.Module.
        constrained_out = x # Regular multi-label training.
        return constrained_out

# Constraint matrix
R_constraint_matrix = torch.zeros([NUMBER_OF_CLASSES, NUMBER_OF_CLASSES])
# Rows indicate a super class (baba), all the columns that are 1 are children of
# the ancestor (super class). This constraint matrix shows the hierarchy.
R_constraint_matrix.fill_diagonal_(1)
for class_name in C_DICT.keys():
    descendants  = list(nx.descendants(DFAS_TREE, class_name))
    ancestor_idx = C_DICT[class_name]
    for child_class_name in descendants:
        child_idx = C_DICT[child_class_name]
        # Normally columns are descendents and rows are ancestors.
        R_constraint_matrix[ancestor_idx, child_idx] = 1
#Transpose to get the ancestors for each node 
R_constraint_matrix = R_constraint_matrix.transpose(1, 0)
R_constraint_matrix = R_constraint_matrix.unsqueeze(dim=0).to(DEVICE)

# Constraint model
single_constraint_eff_net = ConstrainedEffNet(R_constraint_matrix=R_constraint_matrix, eff_net = single_eff_net)
constraint_eff_net = single_constraint_eff_net
# Multi GPU train
# constraint_eff_net = torch.nn.DataParallel(single_constraint_eff_net, device_ids = GPU_IDS)


def main():
    pass



if __name__ == "__main__":
    # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
    # eff_net.eval().to("cuda")
    # uris = [
    #     'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    #     'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    #     'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    #     'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
    # ]
    # batch = torch.cat([utils.prepare_input_from_uri(uri) for uri in uris]).to("cuda")
    # with torch.no_grad():
    #     output = torch.nn.functional.softmax(eff_net(batch), dim = 1)
    # results = utils.pick_n_best(predictions = output, n = 5)
    # for uri, result in zip(uris, results):
    #     img = Image.open(requests.get(uri, stream=True).raw)
    #     img.thumbnail((256,256), Image.ANTIALIAS)
    #     plt.imshow(img)
    #     plt.show()
    #     print(result)
    print(single_constraint_eff_net)
