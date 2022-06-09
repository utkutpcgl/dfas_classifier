import os
import random

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import networkx as nx

from glob import glob


seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

my_tree = nx.DiGraph()
my_tree.add_node('insan')
my_tree.add_node('arac')
my_tree.add_edge('insan', 'askeri')
my_tree.add_edge('askeri', 'silahlı')
my_tree.add_edge('askeri', 'silahsız')
my_tree.add_edge('insan', 'sivil')
my_tree.add_edge('arac', 'sivil arac')
my_tree.add_edge('arac', 'askeri arac')
my_tree.add_edge('askeri arac', 'lastikli')
my_tree.add_edge('askeri arac', 'paletli')
my_tree.add_edge('paletli', 'tank')
my_tree.add_edge('paletli', 'ZMA')
my_tree.add_edge('tank', 'leopard')
my_tree.add_edge('tank', 'm60')
my_tree.add_edge('tank', 'm48')

c_list = {'insan':0, 'arac':1, 'askeri':2, 'sivil':3, 'silahlı':4, 'silahsız':5, \
        'sivil arac':6, 'askeri arac':7, 'lastikli':8, 'paletli':9, 'tank':10, 'ZMA':11,\
        'leopard':12, 'm60':13, 'm48':14}
c_list_keys = list(c_list.keys())

class KGozDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_paths = sorted(glob(os.path.join(images_path, "*")))
        self.transform = transform
    
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        print(self.images_paths[index])
        image = Image.open(self.images_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    constr_r = c_out.clone()
    constr_r[R_batch == 0] = -float('inf')
    final_out, _ = torch.max(constr_r, dim = 2)
    return final_out

class Residual(nn.Module):
    def __init__(self, channel, kernel_size):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size=kernel_size, stride=1, padding=int(kernel_size/2)),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(in_channels = channel, out_channels=channel, kernel_size=kernel_size, stride=1, padding=int(kernel_size/2)),
            nn.BatchNorm2d(channel),
        )
    def forward(self, x):
        identity = x
        result = self.layers(x)
        result += identity
        result = self.relu(result)
        return result

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1):
        super(ConvLayer, self).__init__()  
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        result = self.layers(x)
        return result

class ConstrainedModel(nn.Module):
    def __init__(self, R):
        super(ConstrainedModel, self).__init__()
        self.flatten = nn.Flatten()
        self.R = R
        self.convolutional = nn.Sequential(
            ConvLayer(3,64,5,1,2),
            Residual(64, 5),
            nn.MaxPool2d(kernel_size=2),
            ConvLayer(64,64,3,1,1),
            Residual(64, 3),
            nn.MaxPool2d(kernel_size=2),
            ConvLayer(64,32,5,1,2),
            Residual(32, 3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        )
        self.adaptivePool = nn.AdaptiveMaxPool2d((5,5))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16*15*15, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 15),
        )

    def forward(self, x):
        x = self.convolutional(x)
        #x = self.adaptivePool(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        if self.training:
            constrained_out = x # trains regularly
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Set the hyperparameters
    batch_size = 1
    num_class=15

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
    R = torch.zeros([num_class, num_class])
    R.fill_diagonal_(1)
    for i in c_list.keys():
        descendants = list(nx.descendants(my_tree, i))
        super = c_list[i]
        for child in descendants:
            sub = c_list[child]
            R[super, sub] = 1

    R = R.unsqueeze(0).to(device)
    
    train_data = KGozDataset(
        images_path="crop-tank-images",
        transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ]),
    )

    test_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = ConstrainedModel(R)
    model.to(device)

    # load
    model.load_state_dict(torch.load("../hier_classification_weights.pth"))

    model.eval()

    with torch.no_grad(), open("test_res.txt", "w") as f:
        for _, x in enumerate(test_dataloader):
            x = x.to(device)
            constrained_output = model(x.float())

            predicted = constrained_output.data > 0.6

            res = list(map(int, predicted.squeeze().detach().cpu().numpy()))

            print(
                res,
                file=f
            )
            for i, c in enumerate(res):
                if c == 1:
                    if i > 1:
                        print(" --> ", end='', file=f)
                    print(c_list_keys[i], end='', file=f)
            print("\n", file=f)
        
    print("Done!")


if __name__ == "__main__":
    main()