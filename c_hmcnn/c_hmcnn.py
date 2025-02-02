import os
from networkx.readwrite.json_graph import tree
import torch
from torch import Tensor
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

seed = 5
# NOTE what are these for?
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

max_accuracy = 0
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
        if self.training: # Regular multi-label training.
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_next_exp_num()-> int:
    max_exp_num = 0
    for i in Path(".").iterdir():
        print(type(i))
        content_name = i.stem
        if "exp" in content_name:
            new_exp_num = int(content_name[3:])
            max_exp_num = max(max_exp_num, new_exp_num)
    exp_num = max_exp_num + 1
    return exp_num


def main():
    labels_file=Path("combined_classification_dataset/dataset.txt")
    images_path = Path("combined_classification_dataset/hier_classification_dataset")
    weights_path = Path('hier_classification_weights.pth')
    
    # Calculate experiment number
    exp_num = calculate_next_exp_num()
    exp_folder = Path(f"exp{exp_num}")
    exp_folder.mkdir(exist_ok=True)

    # Set the hyperparameters
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 10
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
        labels_file=labels_file,
        images=images_path,
        transform = transforms.Compose([
            transforms.Resize((120,120)),
            transforms.ToTensor(),
        ]),
    )

    train_size = int(len(train_data)*0.8)
    #splitting data for training and validation
    # lengths = [train_size, len(train_data)-train_size]
    train_split = torch.utils.data.Subset(train_data, torch.arange(train_size))
    valid_split = torch.utils.data.Subset(train_data, torch.arange(start = train_size, end = len(train_data)))
    # train_split, valid_split = torch.utils.data.random_split(train_data, lengths)
    # We should not randomly split the val and train set, since video frames are consecutive.


    train_dataloader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_split, batch_size=batch_size,num_workers=2,pin_memory=True
    )

    model = ConstrainedModel(R)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
    loss_fn = nn.BCEWithLogitsLoss() # Regular multi-label training.

    # Set patience 
    patience, max_patience = 20, 20
    max_score = 0.0
    
    
    train_losses = []
    train_accuracy = []
    train_scores = []
    val_losses = []
    val_accuracy = []
    val_scores = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # TRAIN LOOP
        # -----------------------------------------------------
        total_train = len(train_dataloader.dataset)
        batch_num = len(train_dataloader)
        correct_train = 0
        train_score = 0
        model.train()
        avg_loss=0
        for batch, (X, labels) in enumerate(train_dataloader):
            X = X.to(device)
            labels = labels.to(device)
            # Compute prediction and loss
            output = model(X.float())
            # Penlize the model only for wrong predictions that are part of the esitmated contraint (max probbable class hier) of the model.
            constr_output=get_constr_out(output, R)
            train_output = labels*output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1-labels)*constr_output.double() + labels*train_output # TODO utku this might is the special loss function.
            loss = loss_fn(train_output, labels)
            avg_loss+=loss.item()
            # UTKU no need for sigmoid above, because loss_fn applies to logits.
            # UTKU but below it is necessary.
            predicted=constr_output.data > 0.6
            # total correct predictions
            correct_train += torch.all((predicted == labels.byte()), dim=1).sum()
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad(set_to_none=True)
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{total_train:>5d}]")
        
        # RESULTS PRINTING
        constr_output = constr_output.to('cpu')
        labels = labels.to('cpu')
        avg_loss = float(avg_loss)/float(batch_num)
        train_score = average_precision_score(labels, constr_output.data, average='micro')
        train_losses.append(avg_loss)
        train_scores.append(train_score)
        train_accuracy.append(100*float(correct_train)/float(total_train))
        print(f"Train Error: \n Accuracy: {100*float(correct_train)/float(total_train):>0.1f}%,\
             Train Score: {train_score:>8f} \n")
        # END OF TRAIN LOOP



        # TEST LOOP
        # -------------------------------------------------------        

        model.eval()
        optimizer.zero_grad(set_to_none=True)
        global max_accuracy
        correct = 0
        total=len(valid_dataloader.dataset)
        batch_num = len(valid_dataloader)
        avg_val_loss=0

        with torch.no_grad():
            for batch, (x, y) in enumerate(valid_dataloader):
                x = x.to(device)
                y = y.to(device)
                constrained_output = model(x.float())
                val_loss = loss_fn(constrained_output, y)
                avg_val_loss+=val_loss.item()
                
                # Intern has forgotten to apply sigmoid function to raw logits.
                predicted = constrained_output.sigmoid().data > 0.6
                # Total correct predictions
                correct += torch.all((predicted.byte() == y.byte()), dim=1).sum()

                #Move output and label back to cpu to be processed by sklearn
                cpu_constrained_output = constrained_output.to('cpu')
                y = y.to('cpu')

                if batch == 0:
                    constr_val = cpu_constrained_output
                    y_val = y
                else:
                    constr_val = torch.cat((constr_val, cpu_constrained_output), dim=0)
                    y_val = torch.cat((y_val, y), dim =0)
        
        avg_val_loss = float(avg_val_loss)/float(batch_num)
        accuracy = float(correct)/float(total)
        if accuracy > max_accuracy: # saves only best score.
            max_accuracy = accuracy
            torch.save(model.state_dict(), exp_folder/weights_path)
        score = average_precision_score(y_val, constr_val.data, average='micro')

        val_losses.append(avg_val_loss)
        val_accuracy.append(100*accuracy)
        val_scores.append(score)
        
        print(f"Test Error: \n Accuracy: {(100*accuracy):.1f}%, Precision score: ({score:.5f})\n")
        
        if score >= max_score:
            patience = max_patience
            max_score = score
        else:
            patience = patience - 1

        if patience == 0: break

    print(f"My parameter count: {count_parameters(model)} My highest accuracy: {(100*max_accuracy):.3f}%")
    print("Done!")

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(exp_folder/"Loss", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(val_accuracy,label="val")
    plt.plot(train_accuracy,label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(exp_folder/"Accuracy", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Average Precision Score")
    plt.plot(val_scores,label="val")
    plt.plot(train_scores,label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Average Precision Score")
    plt.legend()
    plt.savefig(exp_folder/"average_precision_score", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()