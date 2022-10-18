from filecmp import DEFAULT_IGNORES
from genericpath import exists
import numpy
import torch
import time
import os
import copy
from tqdm import tqdm
import argparse
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from dataloader import (
    dataloaders,
    dataset_sizes,
    hyps,
    test_dataloader,
    testset_size,
    GPU_ID0,
    GPU_IDS,
    DEVICE,
    TEST_PATH,
    TRAIN_PATH,
    IMG_RES,
    RESIZE,
    TO_TENSOR,
    NORMALIZE,
)
from classifier_net import net

# For testing on a path
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms

# device = torch.device("cuda:0" if torch)
torch.backends.cudnn.benchmark = True
# TODO check if automatic mixed precision reduces accuracy.


def write_log(info_str, path=Path("weights/log.txt")):
    with open(path, "a") as text_file:
        text_file.write(info_str + "\n")


def save_model(model_state_dict, info_str, save_path="weights/eff_weights.pt", log_path="weights/eff_weights_log.txt"):
    if os.path.exists(save_path):
        os.remove(save_path)  # deleting the file
    Path("weights").mkdir(exist_ok=True)
    torch.save(model_state_dict, save_path)
    write_log(info_str, path=log_path)


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
            total_labels, total_preds = [], []
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
                total_labels.append(labels_batch.int().cpu().numpy())
                total_preds.append(preds_batch.int().cpu().numpy())
            if phase == "train":
                scheduler.step()

            # concat over batch dim
            total_labels_tensor = numpy.concatenate(total_labels, axis=0)
            total_preds_tensor = numpy.concatenate(total_preds, axis=0)

            prec = precision_score(total_labels_tensor, total_preds_tensor, pos_label=1, average="macro")
            rec = recall_score(total_labels_tensor, total_preds_tensor, pos_label=1, average="macro")
            f1 = f1_score(total_labels_tensor, total_preds_tensor, pos_label=1, average="macro")
            epoch_acc = accuracy_score(total_labels_tensor, total_preds_tensor)

            # Since last batch might contain less elements averaging over batch_num is incorrect.
            epoch_loss = running_loss / dataset_sizes[phase]
            # Report the new loss
            epoch_report = f"{phase} -> epoch: {epoch}, loss: {epoch_loss}, accuracy: {epoch_acc}, precision: {prec}, recall: {rec}, f1: {f1}"
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


def test_batch(model, test_dataloader, log_path, criterion, testset_size):
    model = model.to(DEVICE)
    print("-" * 10)
    print(f"TEST MODEL:")
    start_timer_epoch = time.time()
    model.eval()  # Switch to eval
    # Get data from dataloader
    running_loss = 0.0
    total_labels, total_preds = [], []
    for input_batch, labels_batch in tqdm(test_dataloader):
        input_batch, labels_batch = input_batch.to(DEVICE), labels_batch.to(DEVICE)
        batch_size = input_batch.size(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision inference.
                output_batch = model(input_batch)
                # calculate loss
                loss = criterion(output_batch, labels_batch)
        preds_batch = torch.argmax(output_batch, dim=1)
        running_loss += loss.item() * batch_size
        corrects = torch.sum(preds_batch == labels_batch.data)
        total_labels.append(labels_batch.int().cpu().numpy())
        total_preds.append(preds_batch.int().cpu().numpy())

    assert len(total_labels) == len(total_preds)
    # concat over batch dim
    total_labels_tensor = numpy.concatenate(total_labels, axis=0)
    total_preds_tensor = numpy.concatenate(total_preds, axis=0)

    prec_1 = precision_score(total_labels_tensor, total_preds_tensor, pos_label=1, average="macro")
    rec_1 = recall_score(total_labels_tensor, total_preds_tensor, pos_label=1, average="macro")
    f1_1 = f1_score(total_labels_tensor, total_preds_tensor, pos_label=1, average="macro")
    prec_0 = precision_score(total_labels_tensor, total_preds_tensor, pos_label=0, average="macro")
    rec_0 = recall_score(total_labels_tensor, total_preds_tensor, pos_label=0, average="macro")
    f1_0 = f1_score(total_labels_tensor, total_preds_tensor, pos_label=0, average="macro")
    accuracy = accuracy_score(total_labels_tensor, total_preds_tensor)

    # Since last batch might contain less elements averaging over batch_num is incorrect.
    loss = running_loss / testset_size
    epoch_time_elapsed = time.time() - start_timer_epoch
    print(f"One epoch for test complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")
    epoch_report = (
        f"TEST -> loss: {loss}, accuracy: {accuracy}\n"
        f"   for pos_label=0 -> precision: {prec_0}, recall: {rec_0}, f1: {f1_0}\n"
        f"   for pos_label=1 -> precision: {prec_1}, recall: {rec_1}, f1: {f1_1}"
    )
    print(epoch_report)
    write_log(epoch_report, path=log_path)


def test(model, test_path, log_path):
    model = model.to(DEVICE)
    print("-" * 10)
    print(f"TEST MODEL:")
    start_timer_epoch = time.time()
    model.eval()  # Switch to eval
    # Get data from dataloader
    label_folders = [str(label) for label in test_path.iterdir() if label.is_dir()]
    class_names = [label.name for label in test_path.iterdir() if label.is_dir()]
    class_names.sort()  # Make sure class index order is correct.
    label_folders.sort()  # Make sure class index order is correct.
    image_paths_list = [
        [img for img in Path(image_folder_path).iterdir() if img.is_file()] for image_folder_path in label_folders
    ]
    epoch_report = ""
    for cls_idx, image_path_list in enumerate(image_paths_list):
        class_name = class_names[cls_idx]
        epoch_report += f"\n\n\n\n\n{class_name}\n".upper()
        for img_path in tqdm(image_path_list):
            # read image
            pil_img = Image.open(img_path)
            pil_img = RESIZE(pil_img)  # can resize pil :)
            torch_img = TO_TENSOR(pil_img).cuda()
            torch_img = NORMALIZE(torch_img)
            # run model on image
            with torch.no_grad():
                # check inference and ground truth and report it to a log file.
                with torch.cuda.amp.autocast():  # Mixed precision inference.
                    output = model(torch_img.unsqueeze(dim=0))
            pred = torch.argmax(output, dim=1)  # First dim is batch dimension
            is_accurate: bool = int(pred.item()) == cls_idx
            epoch_report += f"{img_path} is predicted {is_accurate*'correctly' or 'falsely'}\n"
        # Write results to different log paths.
    write_log(epoch_report, path=log_path)
    epoch_time_elapsed = time.time() - start_timer_epoch
    print(f"One epoch for test complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")


def main(weight_path: str, log_path: Path, test_bool: bool, test_weight_path: str, test_batch_bool: bool):
    """The training loop all pieces combined. https://towardsdatascience.com/why-adamw-matters-736223f31b5d
    Researchers often prefer stochastic gradient descent (SGD) with momentum because models trained with Adam have been observed to not generalize as well."""
    epochs = hyps["epochs"]
    optimizer = torch.optim.AdamW(net.parameters(), lr=hyps["lr"])
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1, 10]).float().to(DEVICE)
    )  # Give more importance to doğrultmuş.
    # TODO cosineannealinglr scheduler can be used for better performance.
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=hyps["lr_step_size"], gamma=hyps["lr_gamma"]
    )
    try:
        if test_bool:
            net.load_state_dict(torch.load(test_weight_path))
            test(model=net, test_path=TEST_PATH, log_path=log_path)
        elif test_batch_bool:
            net.load_state_dict(torch.load(test_weight_path))
            test_batch(
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
            file_suffix = content_name.split("_")[-1]
            if file_suffix.isdigit():
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
    task = hyps["TASK"]
    weight_name = f"{task}_yonelim_classifier_{hyps['MODEL']}"
    parser.add_argument("--weight_path", type=str, default=f"{weight_name}.pt", help="weights path")
    # NOTE store_false is default true while store_true is default false.
    parser.add_argument("--test", action="store_true", help="Test on testset.")
    parser.add_argument("--test_batch", action="store_true", help="Test on testset with batches to see metrics.")
    parser.add_argument(
        "--test_weight_path",
        type=str,
        default=f"exp_{task}_yonelim_classifier_resnet18_1/{task}_yonelim_classifier_resnet18.pt",
        help="weights path",
    )
    opt = parser.parse_args()
    test_bool = opt.test
    test_batch_bool = opt.test_batch
    # Default option is train
    if not (test_bool or test_batch_bool):
        test_weight_path = None
        exp_folder = create_exp_folder(exp_name=f"exp_{test_bool * 'test_'}{weight_name}")
        weight_path = exp_folder / (test_bool * "test_" + opt.weight_path)
        log_path = exp_folder / (weight_path.stem + "_log.txt")
    else:
        test_weight_path = opt.test_weight_path
        weight_path = None
        test_type = "test_batch" * test_batch_bool or "test"
        log_path = Path(test_weight_path).with_name(f"{test_type}.log")
    # Add test to log path if if test_bool.
    main(weight_path, log_path, test_bool, test_weight_path, test_batch_bool)
