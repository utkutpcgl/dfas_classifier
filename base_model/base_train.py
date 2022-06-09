import argparse
import torch
import copy
import time
from pathlib import Path
import os
from tqdm import tqdm
from base_model import base_net, GPU_ID0
from base_dataloader import dataloaders, dataset_sizes, hyps
import shutil


# device = torch.device("cuda:0" if torch)
DEVICE = f"cuda:{GPU_ID0}"
torch.backends.cudnn.benchmark = True


def train(model, dataloaders_val_train, criterion, optimizer, scheduler, epochs, weight_name):
    model = model.to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    Path("logs/weights/").mkdir(parents=True, exist_ok=True)
    shutil.copyfile("hyperparams.yaml", f"logs/weights/{weight_name}.yaml")
    best_weights = copy.deepcopy(model.state_dict())
    save_model(
        best_weights,
        info_str="first_log",
        save_path=f"logs/weights/{weight_name}.pt",
        log_path=f"logs/{weight_name}.txt",
    )
    start_timer_total = time.time()
    best_acc = 0.0
    for epoch in range(epochs):
        print("-" * 10)
        print(f"Epoch {epoch}/{epochs - 1}")
        for phase in ["train", "val"]:
            start_timer_epoch = time.time()
            # Switch btw eval or train mode
            model.train() if phase == "train" else model.eval()
            # Get data from dataloader
            running_loss = 0.0
            running_corrects = 0
            for input_batch, labels_batch in tqdm(dataloaders_val_train[phase]):
                input_batch, labels_batch = input_batch.to(DEVICE), labels_batch.to(DEVICE)
                batch_size = input_batch.size(0)
                # zero gradients to avoid accumulation
                optimizer.zero_grad()
                # Track history only for training
                with torch.set_grad_enabled(phase == "train"):
                    # run model on data
                    with torch.cuda.amp.autocast():
                        # For automatic mixed precision.
                        output_batch = model(input_batch)
                        # calculate loss
                        loss = criterion(output_batch, labels_batch)
                    preds_batch = torch.argmax(output_batch, dim=1)

                    # run backprop and update weights in train phase only
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    # update running loss
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds_batch == labels_batch.data)
            if phase == "train":
                scheduler.step()
            # Since last batch might contain less elements averaging over batch_num is incorrect.
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # Report the new loss
            epoch_report = f"{phase} -> epoch: {epoch}, loss: {epoch_loss}, accuracy: {epoch_acc}"
            print(epoch_report)
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                save_model(
                    best_weights,
                    info_str=epoch_report,
                    save_path=f"logs/weights/{weight_name}.pt",
                    log_path=f"logs/{weight_name}.txt",
                )
            epoch_time_elapsed = time.time() - start_timer_epoch
            print(f"One epoch for {phase} complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")

    total_time_elapsed = time.time() - start_timer_total
    print(f"Training complete in {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")
    save_model(best_weights, save_path=f"logs/weights/{weight_name}.pt", log_path=f"logs/{weight_name}.txt")
    return model


def write_log(info_str, path):
    with open(path, "a") as text_file:
        text_file.write(info_str + "\n")


def save_model(model_state_dict, info_str, save_path, log_path):
    if os.path.exists(save_path):
        os.remove(save_path)  # deleting the file
    torch.save(model_state_dict, save_path)
    write_log(info_str, path=log_path)


def main(weight_name: str):
    """The training loop all pieces combined. https://towardsdatascience.com/why-adamw-matters-736223f31b5d
    Researchers often prefer stochastic gradient descent (SGD) with momentum because models trained with Adam have been observed to not generalize as well."""
    epochs = hyps["epochs"]
    # Do not optimize parameters without gradient.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_net.parameters()), lr=hyps["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    # TODO cosineannealinglr scheduler can be used for better performance.
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=hyps["lr_step_size"], gamma=hyps["lr_gamma"]
    )
    try:
        train(
            base_net,
            dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            epochs=epochs,
            weight_name=weight_name,
        )
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        print("Cleared cuda cache, bye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_name", type=str, default=hyps["NAME"], help="weights name")
    opt = parser.parse_args()
    main(opt.weight_name)
