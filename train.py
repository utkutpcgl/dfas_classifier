from filecmp import DEFAULT_IGNORES
from genericpath import exists
import torch
import time
import os
import copy
from tqdm import tqdm
import argparse
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from dataloader import dataloaders, dataset_sizes, hyps, test_dataloader, testset_size, GPU_ID0, GPU_IDS, DEVICE
from classifier_net import net

# device = torch.device("cuda:0" if torch)
torch.backends.cudnn.benchmark = True
# TODO check if automatic mixed precision reduces accuracy.


def train(model, dataloaders_val_train, criterion, optimizer, scheduler, epochs, weight_path, log_path):
    model = model.to(DEVICE)
    # To avoid gradient underflowing due to fp16 training (mixed precision)
    # NOTE scales gradients (increases) for loss calculation and under scales before updating
    scaler = torch.cuda.amp.GradScaler()
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch_report = ""
    save_model(best_weights, info_str="first_log", save_path=weight_path, log_path=log_path)
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
                    # run model on data with mixed precision
                    with torch.cuda.amp.autocast():
                        output_batch = model(input_batch)
                        # calculate loss
                        loss = criterion(output_batch, labels_batch)

                    preds_batch = torch.argmax(output_batch, dim=1)

                    # run backprop and update weights in train phase only
                    if phase == "train":
                        # 1. Scale loss -> calculate scaled gradients. Gradient scaling improves convergence for
                        # networks with float16 gradients by minimizing gradient underflow
                        scaler.scale(loss).backward()
                        # 2. First unscale gradients (auto if unscale not called), then update weights
                        scaler.step(optimizer)
                        # 3. Choose how much to scale smartly.
                        scaler.update()
                        # Normal methods -> 1. loss.backward() & 2. optimizer.step()
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
                best_epoch_report = epoch_report
                save_model(best_weights, info_str=best_epoch_report, save_path=weight_path, log_path=log_path)
            epoch_time_elapsed = time.time() - start_timer_epoch
            print(f"One epoch for {phase} complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")

    total_time_elapsed = time.time() - start_timer_total
    print(f"Training complete in {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")
    save_model(best_weights, info_str=best_epoch_report, save_path=weight_path, log_path=log_path)
    return model


def test(model, test_dataloader, log_path, criterion, testset_size):
    model = model.to(DEVICE)
    print("-" * 10)
    print(f"TEST MODEL:")
    start_timer_epoch = time.time()
    model.eval()  # Switch to eval
    # Get data from dataloader
    running_loss = 0.0
    running_corrects = 0
    for input_batch, labels_batch in tqdm(test_dataloader):
        input_batch, labels_batch = input_batch.to(DEVICE), labels_batch.to(DEVICE)
        batch_size = input_batch.size(0)
        with torch.cuda.amp.autocast():  # Mixed precision inference.
            output_batch = model(input_batch)
            # calculate loss
            loss = criterion(output_batch, labels_batch)
        preds_batch = torch.argmax(output_batch, dim=1)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds_batch == labels_batch.data)

    # Since last batch might contain less elements averaging over batch_num is incorrect.
    loss = running_loss / testset_size
    acc = running_corrects.double() / testset_size
    # TODO add f1 score.
    epoch_time_elapsed = time.time() - start_timer_epoch
    print(f"One epoch for test complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")
    epoch_report = f"TEST -> loss: {loss}, accuracy: {acc}"
    print(epoch_report)
    write_log(epoch_report, path=log_path)
    return model


def write_log(info_str, path="weights/log.txt"):
    with open(path, "a") as text_file:
        text_file.write(info_str + "\n")


def save_model(model_state_dict, info_str, save_path="weights/eff_weights.pt", log_path="weights/eff_weights_log.txt"):
    if os.path.exists(save_path):
        os.remove(save_path)  # deleting the file
    Path("weights").mkdir(exist_ok=True)
    torch.save(model_state_dict, save_path)
    write_log(info_str, path=log_path)


def main(weight_path: str, log_path: str, test_bool: bool, test_weight_path: str):
    """The training loop all pieces combined. https://towardsdatascience.com/why-adamw-matters-736223f31b5d
    Researchers often prefer stochastic gradient descent (SGD) with momentum because models trained with Adam have been observed to not generalize as well."""
    epochs = hyps["epochs"]
    optimizer = torch.optim.AdamW(net.parameters(), lr=hyps["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    # TODO cosineannealinglr scheduler can be used for better performance.
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=hyps["lr_step_size"], gamma=hyps["lr_gamma"]
    )
    try:
        if test_bool:
            net.load_state_dict(torch.load(test_weight_path))
            test(
                model=net,
                test_dataloader=test_dataloader,
                log_path=log_path,
                criterion=criterion,
                testset_size=testset_size,
            )
        else:
            train(
                net,
                dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=exp_lr_scheduler,
                epochs=epochs,
                weight_path=weight_path,
                log_path=log_path,
            )
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        print("Cleared cuda cache, bye!")


def calculate_next_exp_num(exp_name: str) -> int:
    # Calculate experiment number
    max_exp_num = 0
    for i in Path(".").iterdir():
        content_name = i.name
        if exp_name in content_name:
            old_exp_num = int(content_name.split("_")[-1])
            max_exp_num = max(max_exp_num, old_exp_num)
    exp_num = max_exp_num + 1
    return exp_num


def create_exp_folder(exp_name, hyps_path=Path("hyperparameters.yaml")) -> Path:
    exp_num = calculate_next_exp_num(exp_name)
    exp_folder = Path(f"{exp_name}_{exp_num}")
    exp_folder.mkdir(exist_ok=True)
    copyfile(hyps_path, exp_folder / hyps_path.name)
    return exp_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    weight_name = f"mevzi_classifier_{hyps['MODEL']}"
    parser.add_argument("--weight_path", type=str, default=f"{weight_name}.pt", help="weights path")
    # NOTE store_false is default true while store_true is default false.
    parser.add_argument("--test", action="store_true", help="save cropped prediction boxes")
    parser.add_argument(
        "--test_weight_path",
        type=str,
        default=f"exp_mevzi_classifier_resnet_1/mevzi_classifier_resnet.pt",
        help="weights path",
    )
    opt = parser.parse_args()
    test_bool = opt.test
    exp_folder = create_exp_folder(exp_name=f"exp_{test_bool * 'test_'}{weight_name}")
    test_weight_path = opt.test_weight_path
    weight_path = exp_folder / (test_bool * "test_" + opt.weight_path)
    # Add test to log path if if test_bool.
    log_path = exp_folder / (weight_path.stem + "_log.txt")
    print(log_path)
    main(weight_path, log_path, test_bool, test_weight_path)
