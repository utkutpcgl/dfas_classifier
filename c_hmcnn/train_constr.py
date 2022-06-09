from filecmp import DEFAULT_IGNORES
from genericpath import exists
import torch
import time
import os
import copy
from dataloader import dataloaders, dataset_sizes, hyps
from tqdm import tqdm
import argparse
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from dataloader import GPU_ID0,GPU_IDS,DEVICE
from c_hmcnn_efficient_net import R_constraint_matrix, get_constr_out, constraint_eff_net

# device = torch.device("cuda:0" if torch)
torch.backends.cudnn.benchmark = True


def train_constraint(model, dataloaders_val_train, criterion, optimizer, scheduler, epochs, weight_path, log_path):
    model = model.to(DEVICE)
    # To avoid gradient underflowing due to fp16 training (mixed precision)
    # NOTE scales gradients (increases) for loss calculation and under scales before updating
    scaler = torch.cuda.amp.GradScaler()  
    best_weights = copy.deepcopy(model.state_dict())
    save_model(best_weights, info_str="first_log", save_path=weight_path, log_path=log_path)
    start_timer_total = time.time()
    best_acc = 0.0
    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{epochs - 1}')
        for phase in ["train", "val"]:
            start_timer_epoch = time.time()
            # Switch btw eval or train mode
            if phase == "train":
                model.train()
            else:
                model.eval()
            # Get data from dataloader
            running_loss = 0.0
            running_corrects_specific = 0
            running_corrects_hier = 0
            running_falses_hier = 0
            running_labels_hier = 0
            for input_batch, labels_batch in tqdm(dataloaders_val_train[phase]):
                input_batch, labels_batch = input_batch.to(DEVICE), labels_batch.to(DEVICE)
                batch_size = input_batch.size(0)
                # Convert labels to n-hot vectors.
                # print(R_constraint_matrix)
                # print(R_constraint_matrix.shape)
                # print(labels_batch[0:5])
                # R_constr_matrix is 3 d array with, the column will give all parent + current class idx.
                # BUG there probably is a indexing mistake below, or the matrix is not what I want.
                n_hot_labels = R_constraint_matrix[0, labels_batch, :] # Get ancestors of labels as n-hot.
                # print(n_hot_labels.shape)
                # print(n_hot_labels[0:5])

                # zero gradients to avoid accumulation
                optimizer.zero_grad()
                # Track history only for training
                with torch.set_grad_enabled(phase=="train"):
                    # run model on data with mixed precision
                    with torch.cuda.amp.autocast():
                        # print(input_batch.shape)
                        output_batch = model(input_batch)
                        # # NOTE the special loss function seems to be useless (not producing meaningful values.)
                        # # It did not affect accuracy in a positive manner.
                        # # Calculate the logit values only inside the label constraint and prediction constraint
                        # # NOTEDirectly penalizing the model for whole class predictions might also work??
                        # constr_output = get_constr_out(output_batch, R_constraint_matrix)
                        # selected_output = n_hot_labels*output_batch.double()
                        # constr_selected_output = get_constr_out(selected_output, R_constraint_matrix)
                        # # Penlize the model only for wrong predictions that are part of the esitmated contraint (max probbable class hier) of the model.
                        # pred_label_combined_constr_output = (1-n_hot_labels)*constr_output.double() + n_hot_labels*constr_selected_output # TODO utku this might is the special loss function.
                        # # calculate loss
                        # # print("pred_label_combined_constr_output: ",pred_label_combined_constr_output.shape)
                        # # print(n_hot_labels.shape)
                        # # BCEWITHLOGITSLOSS takes care of sigmoid (takes only logits as input) and applies cross entropy loss.
                        # loss = criterion(pred_label_combined_constr_output, n_hot_labels)
                        # TODO you have to add specail dataset (dataloader) to read n-hot labels for each image.
                        loss = criterion(output_batch, n_hot_labels)

                    preds_batch = torch.argmax(output_batch, dim = 1) # For normal accuracy calculation
                    # n_hot_preds = constr_output > 0.3 #prev 0.6
                    n_hot_preds = torch.sigmoid(output_batch) > 0.7 #prev 0.6
                    
                    # run backprop and update weights in train phase only
                    if phase=="train":
                        # 1. Scale loss -> calculate scaled gradients
                        scaler.scale(loss).backward() 
                        # 2. First unscale gradients (auto if unscale not called), then update weights
                        scaler.step(optimizer)
                        # 3. Choose how much to scale smartly.
                        scaler.update()
                        scheduler.step()
                        # Normal methods -> 1. loss.backward() & 2. optimizer.step()
                    # update running loss running_falses_hier
                running_loss += loss.item() * batch_size
                running_corrects_specific += torch.sum(preds_batch == labels_batch)

                running_corrects_hier += torch.sum(torch.logical_and(n_hot_preds, n_hot_labels))
                running_falses_hier += torch.sum(n_hot_preds != n_hot_labels)
                running_labels_hier += torch.sum(n_hot_labels)
            # Since last batch might contain less elements averaging over batch_num is incorrect.
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc_specific = running_corrects_specific.double() / dataset_sizes[phase] # Calculate hard accuracy.
            epoch_acc_hier = (running_corrects_hier - running_falses_hier) / running_labels_hier # Calculate soft accuracy.
            # Report the new loss
            epoch_report = f"{phase} -> epoch: {epoch}, loss: {epoch_loss}, accuracy_hier: {epoch_acc_hier}"
            print(epoch_report)
            if phase == "val" and epoch_acc_hier > best_acc:
                best_acc = epoch_acc_hier
                best_weights = copy.deepcopy(model.state_dict())
                save_model(best_weights, info_str=epoch_report, save_path=weight_path, log_path = log_path)
            epoch_time_elapsed = time.time() - start_timer_epoch
            print(f'One epoch for {phase} complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')
        
    total_time_elapsed = time.time() - start_timer_total
    print(f'Training complete in {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    return model
            

def write_log(info_str, path = "weights/log.txt"):
    with open(path,"a") as text_file:
        text_file.write(info_str + "\n")

def save_model(model_state_dict ,info_str, save_path='weights/eff_weights.pt', log_path = "weights/eff_weights_log.txt"):
    if os.path.exists(save_path): 
        os.remove(save_path)  # deleting the file
    Path("weights").mkdir(exist_ok = True)
    torch.save(model_state_dict,save_path)
    write_log(info_str, path = log_path)


def main(weight_path:str, log_path:str):
    """The training loop all pieces combined. https://towardsdatascience.com/why-adamw-matters-736223f31b5d 
    Researchers often prefer stochastic gradient descent (SGD) with momentum because models trained with Adam have been observed to not generalize as well."""
    epochs = hyps["epochs"]
    optimizer = torch.optim.AdamW(constraint_eff_net.parameters(),lr=hyps["lr"])
    criterion_with_constraint = torch.nn.BCEWithLogitsLoss() # Regular multi-label training.
    # TODO cosineannealinglr scheduler can be used for better performance.
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = hyps["lr_step_size"], gamma = hyps["lr_gamma"])
    try:
        train_constraint(constraint_eff_net, dataloaders, criterion=criterion_with_constraint, optimizer=optimizer,scheduler=exp_lr_scheduler,epochs=epochs, weight_path=weight_path, log_path=log_path)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        print("Cleared cuda cache, bye!")

def calculate_next_exp_num()-> int:
    # Calculate experiment number
    max_exp_num = 0
    for i in Path(".").iterdir():
        content_name = i.stem
        if "constr_ex" in content_name:
            new_exp_num = int(content_name[len("constr_ex"):])
            max_exp_num = max(max_exp_num, new_exp_num)
    exp_num = max_exp_num + 1
    return exp_num

def create_exp_folder(hyps_path = Path("hyperparameters.yaml"))->Path:
    exp_num = calculate_next_exp_num()
    exp_folder = Path(f"constr_ex{exp_num}")
    exp_folder.mkdir(exist_ok=True)
    copyfile(hyps_path, exp_folder/hyps_path.name)
    return exp_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    exp_folder = create_exp_folder()
    weight_name = "effnet"
    parser.add_argument('--weight_path', type=str, default=f"{weight_name}.pt", help='weights path')
    opt = parser.parse_args()
    weight_path = exp_folder / opt.weight_path
    log_path = exp_folder / (Path(opt.weight_path).stem + "_log.txt")
    main(weight_path, log_path)
