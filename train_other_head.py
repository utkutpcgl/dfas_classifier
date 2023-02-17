from pathlib import Path
import time
import torch
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import numpy

from classifier_net import net, NUMBER_OF_CLASSES
from train import DEVICE, write_log

from torchvision import datasets, transforms
from dataloader import (
    DATA_DIR,
    HYPS,
    DEVICE,
    data_transforms,
    DATALOADERS,
    image_datasets,
)
import numpy as np

TRAINED_WEIGHT_PATH = Path(
    "/home/utku/Documents/repos/dfas_classifier/trained_models/exp_light_dataset_v1_classifier_resnet18_1/light_dataset_v1_classifier_resnet18_f1.pt"
)
PRETRAINED_MODEL = False  # can be true if +1 neuron added trained model is available.
# device = torch.device("cuda:0" if torch)
SEED = 1
GET_ENERGY = False
POSSIBLE_SIG_THRESH_TENSOR = torch.tensor([i / 20 for i in range(20)]).to(DEVICE)


def print_and_log(message, log_path: str):
    print(message)
    with open(log_path, "a") as message_appender:
        message_appender.write(message + "\n")


def get_trained_model(weight_path):
    net.load_state_dict(torch.load(weight_path))
    return net


def get_dataloaders():
    data_dir = DATA_DIR / "OOD"
    out_image_datasets = {d_type: datasets.ImageFolder(data_dir / d_type, data_transforms[d_type]) for d_type in ["train", "val"]}
    # pin_memory=True to speed up host to device data transfer with page-locked memory
    out_dataloaders = {
        d_type: torch.utils.data.DataLoader(
            out_image_datasets[d_type],
            batch_size=HYPS["other_head_batch_size_OD"],
            shuffle=True,
            num_workers=HYPS["workers"],
            pin_memory=True,
        )
        for d_type in ["train", "val"]
    }
    in_dataloaders = {
        d_type: torch.utils.data.DataLoader(
            image_datasets[d_type],
            batch_size=HYPS["other_head_batch_size_ID"],
            shuffle=True,
            num_workers=HYPS["workers"],
            pin_memory=True,
        )
        for d_type in ["train", "val"]
    }
    return out_dataloaders, in_dataloaders


def train_one_epoch(model, dataloader_train_in, dataloader_train_out, scheduler, optimizer, state_report):
    model.train()  # enter train mode
    loss_avg = 0.0
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    total_tqdms = len(dataloader_train_out)
    for in_set, out_set in tqdm(zip(dataloader_train_in, dataloader_train_out), total=total_tqdms):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        ood_in_target = torch.ones(size=(len(in_set[1]), 1)).to(DEVICE).squeeze()
        ood_out_target = torch.zeros(size=(len(out_set[1]), 1)).to(DEVICE).squeeze()
        data, target = data.to(DEVICE), target.to(DEVICE)
        # forward
        output = model(data)
        # backward
        optimizer.zero_grad()
        # loss = torch.nn.functional.cross_entropy(output[: len(in_set[0])], target)
        loss_classification = torch.nn.functional.cross_entropy(output[: len(in_set[0]), : (NUMBER_OF_CLASSES - 1)], target)
        assert output[: len(in_set[0]), (NUMBER_OF_CLASSES - 1)].shape == ood_in_target.shape
        loss_ood_in = torch.nn.functional.binary_cross_entropy_with_logits(output[: len(in_set[0]), (NUMBER_OF_CLASSES - 1)], ood_in_target)
        # NOTE torch.nn.functional.binary_cross_entropy does not input logits but sigmoid outputs.
        assert output[len(in_set[0]) :, (NUMBER_OF_CLASSES - 1)].shape == ood_out_target.shape
        loss_ood_out = torch.nn.functional.binary_cross_entropy_with_logits(output[len(in_set[0]) :, (NUMBER_OF_CLASSES - 1)], ood_out_target)
        # Normalize based on sample counts (to increase weights of the one with more samples.)
        loss_ood_normalized = (loss_ood_in * len(ood_in_target) + loss_ood_out * len(ood_out_target)) / (len(ood_in_target) + len(ood_out_target))

        # number_of_tasks = 2
        task_normalized_weights = torch.nn.functional.softmax(torch.randn(2), dim=-1).to(DEVICE)
        loss = loss_classification  # task_normalized_weights[0] * loss_classification + task_normalized_weights[1] * loss_ood_normalized
        loss.backward()
        optimizer.step()
        scheduler.step()  # update the steps of cosine annealing every step (over batches.)
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state_report["avg_train_loss"] = loss_avg


def get_perf_metrics(total_labels_array, total_preds_array):
    # 'macro': Calculate metrics for each label, and find their unweighted mean
    assert total_labels_array.shape == total_preds_array.shape
    epoch_prec = precision_score(total_labels_array, total_preds_array, pos_label=1, average="macro")
    epoch_rec = recall_score(total_labels_array, total_preds_array, pos_label=1, average="macro")
    epoch_f1 = f1_score(total_labels_array, total_preds_array, pos_label=1, average="macro")
    epoch_acc = accuracy_score(total_labels_array, total_preds_array)
    return epoch_prec, epoch_rec, epoch_f1, epoch_acc


def validate_one_epoch(model, dataloader_val_in, dataloader_val_out, state_report):
    model.eval()
    total_loss = 0.0
    total_labels_OOD, total_preds_OOD = [], []
    total_labels_classification, total_preds_classification = [], []
    number_of_thresholds = len(POSSIBLE_SIG_THRESH_TENSOR)
    with torch.no_grad():
        # In distribution iteration for classificaiton and in-dist training
        for input_batch, labels_batch in dataloader_val_in:
            input_batch, labels_batch = input_batch.to(DEVICE), labels_batch.to(DEVICE)
            ood_in_labels_batch = torch.ones(size=(len(input_batch), 1)).to(DEVICE).squeeze()
            # forward
            output_batch = model(input_batch)
            loss_classification = torch.nn.functional.cross_entropy(output_batch[:, :-1], labels_batch)  # Default is mean reduction.
            loss_ood_in = torch.nn.functional.binary_cross_entropy_with_logits(output_batch[:, -1], ood_in_labels_batch)  # Default is mean reduction.
            # classification accuracy
            preds_batch_classification = torch.argmax(output_batch[:, : (NUMBER_OF_CLASSES - 1)], dim=1)
            # To numpy and CPU.
            total_preds_classification.append(preds_batch_classification.int().cpu().numpy())
            total_labels_classification.append(labels_batch.int().cpu().numpy())

            # OOD
            broadcasted_ood_in_labels_batch = torch.broadcast_to(ood_in_labels_batch.unsqueeze(dim=1), (len(output_batch), number_of_thresholds))
            broadcasted_output_batch = torch.broadcast_to(
                output_batch[:, (NUMBER_OF_CLASSES - 1)].unsqueeze(dim=1),
                (len(output_batch), number_of_thresholds),
            )
            broadcasted_sig_thresholds = torch.broadcast_to(POSSIBLE_SIG_THRESH_TENSOR.unsqueeze(dim=0), (len(output_batch), number_of_thresholds))
            broadcasted_predictions_OD = (broadcasted_output_batch > broadcasted_sig_thresholds) * 1
            total_labels_OOD.append(broadcasted_ood_in_labels_batch.int().cpu().numpy())
            total_preds_OOD.append(broadcasted_predictions_OD.int().cpu().numpy())
            # test loss average
            total_loss += float(loss_classification.data) + float(loss_ood_in.data)
        # Out distribution training
        for input_batch, labels_batch in dataloader_val_out:
            input_batch, labels_batch = input_batch.to(DEVICE), labels_batch.to(DEVICE)
            ood_out_labels_batch = torch.zeros(size=(len(input_batch), 1)).to(DEVICE).squeeze()
            # forward
            output_batch = model(input_batch)
            loss_ood_out = torch.nn.functional.binary_cross_entropy_with_logits(
                output_batch[:, -1], ood_out_labels_batch
            )  # Default is mean reduction.

            # OOD
            broadcasted_ood_out_labels_batch = torch.broadcast_to(ood_out_labels_batch.unsqueeze(dim=1), (len(output_batch), number_of_thresholds))
            broadcasted_output_batch = torch.broadcast_to(
                output_batch[:, (NUMBER_OF_CLASSES - 1)].unsqueeze(dim=1),
                (len(output_batch), number_of_thresholds),
            )
            broadcasted_sig_thresholds = torch.broadcast_to(POSSIBLE_SIG_THRESH_TENSOR.unsqueeze(dim=0), (len(output_batch), number_of_thresholds))
            broadcasted_predictions_OD = (broadcasted_output_batch > broadcasted_sig_thresholds) * 1
            total_labels_OOD.append(broadcasted_ood_out_labels_batch.int().cpu().numpy())
            total_preds_OOD.append(broadcasted_predictions_OD.int().cpu().numpy())
            # TODO add f1 , precision and recall per task. Pick the weights with best average f1.
            # test loss average
            total_loss += float(loss_ood_out.data)
    # concat over batch dim
    total_labels_array_classification = numpy.concatenate(total_labels_classification, axis=0)
    total_preds_array_classification = numpy.concatenate(total_preds_classification, axis=0)
    epoch_prec_classification, epoch_rec_classification, epoch_f1_classification, epoch_acc_classification = get_perf_metrics(
        total_labels_array_classification, total_preds_array_classification
    )

    # concat over batch dim, transpose over thresholds dim to iterate over different thresholds.
    total_labels_OOD_array = numpy.concatenate(total_labels_OOD, axis=0).T
    total_preds_OOD_array = numpy.concatenate(total_preds_OOD, axis=0).T
    total_prec_OOD, total_rec_OOD, total_f1_OOD, total_acc_OOD = 0, 0, 0, 0
    # add metrics for different threshold values  (then average them.)
    for total_label_OOD_for_thresh, total_preds_OOD_for_thresh in zip(total_labels_OOD_array, total_preds_OOD_array):
        thresh_prec_OOD, thresh_rec_OOD, thresh_f1_OOD, thresh_acc_OOD = get_perf_metrics(total_label_OOD_for_thresh, total_preds_OOD_for_thresh)
        total_prec_OOD += thresh_prec_OOD
        total_rec_OOD += thresh_rec_OOD
        total_f1_OOD += thresh_f1_OOD
        total_acc_OOD += thresh_acc_OOD
    avg_prec_OOD, avg_rec_OOD, avg_f1_OOD, avg_acc_OOD = (
        total / number_of_thresholds for total in (total_prec_OOD, total_rec_OOD, total_f1_OOD, total_acc_OOD)
    )
    # avg_prec_OOD, avg_rec_OOD, avg_f1_OOD, avg_acc_OOD = 0, 0, 0, 0
    state_report["in_avg_val_loss"] = total_loss / len(dataloader_val_in)
    state_report["in_val_accuracy_classification"] = 100 * epoch_acc_classification
    state_report["in_val_f1_classification"] = 100 * epoch_f1_classification
    state_report["in_val_prec_classification"] = 100 * epoch_prec_classification
    state_report["in_val_rec_classification"] = 100 * epoch_rec_classification
    state_report["out_avg_val_precision"] = 100 * avg_prec_OOD
    state_report["out_avg_val_recall"] = 100 * avg_rec_OOD
    state_report["out_avg_val_f1"] = 100 * avg_f1_OOD
    state_report["out_avg_val_accuracy"] = 100 * avg_acc_OOD
    state_report["final_avg_f1"] = 100 * (avg_f1_OOD + epoch_f1_classification) / 2


def train_oh(
    model,
    dataloader_train_in,
    dataloader_train_out,
    dataloader_val_in,
    dataloader_val_out,
    optimizer,
    scheduler,
    epochs,
    weight_path: Path,
    log_path,
):
    """Fine tune an already trained model to detect out-of-dist samples (other class)."""
    print("Start other head training")
    state_report = {}
    best_report_str = ""
    final_avg_f1 = 0
    for epoch in range(epochs):
        state_report["epoch"] = epoch
        begin_epoch = time.time()
        if epoch == 0:
            print("TEST VALIDATION CODE")
            validate_one_epoch(model=model, dataloader_val_in=dataloader_val_in, dataloader_val_out=dataloader_val_out, state_report=state_report)
        train_one_epoch(
            model,
            dataloader_train_in=dataloader_train_in,
            dataloader_train_out=dataloader_train_out,
            scheduler=scheduler,
            optimizer=optimizer,
            state_report=state_report,
        )
        validate_one_epoch(model=model, dataloader_val_in=dataloader_val_in, dataloader_val_out=dataloader_val_out, state_report=state_report)
        state_report_str = "Epoch {:3d} | Time {:5d} | Train Loss {:.4f} | In Avg. Val Loss {:.3f} | In Val Accuracy Classification {:.2f} | In Val F1 Classification {:.2f} | Out Avg Val Prec:{:.2f} Rec:{:.2f} F1:{:.2f} Acc:{:.2f} | Final avg f1:{:.2f}".format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state_report["avg_train_loss"],
            state_report["in_avg_val_loss"],
            state_report["in_val_accuracy_classification"],
            state_report["in_val_f1_classification"],
            state_report["out_avg_val_precision"],
            state_report["out_avg_val_recall"],
            state_report["out_avg_val_f1"],
            state_report["out_avg_val_accuracy"],
            state_report["final_avg_f1"],
        )
        if final_avg_f1 < state_report["final_avg_f1"]:
            # Save model
            best_report_str = state_report_str
            final_avg_f1 = state_report["final_avg_f1"]
            torch.save(model.state_dict(), weight_path.with_name(weight_path.stem + "_other_head.pt"))
        print_and_log(message=state_report_str + "\n", log_path=log_path)
        if epoch == epochs - 1:
            print_and_log(message="\nBest epoch report : \n", log_path=log_path)
            print_and_log(message=best_report_str + "\n\n\n", log_path=log_path)


def train_other_head(model, out_dataloaders, in_dataloaders):
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPS["other_head_initial_lr"])
    if len(out_dataloaders["train"]) > len(in_dataloaders["train"]):
        total_num_steps = len(in_dataloaders["train"])
    else:
        total_num_steps = len(out_dataloaders["train"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_num_steps, eta_min=HYPS["other_head_final_lr"])
    log_path = TRAINED_WEIGHT_PATH.with_name(TRAINED_WEIGHT_PATH.stem + "_other_head_train_log.txt")
    train_oh(
        model=model,
        dataloader_train_in=in_dataloaders["train"],
        dataloader_train_out=out_dataloaders["train"],
        dataloader_val_in=in_dataloaders["val"],
        dataloader_val_out=out_dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=HYPS["other_head_epochs"],
        weight_path=TRAINED_WEIGHT_PATH,
        log_path=log_path,
    )


if __name__ == "__main__":
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(HYPS["CUDA_FRACTION"], device=DEVICE)
    print("POSSIBLE_SIG_THRESH_TENSOR is: ", POSSIBLE_SIG_THRESH_TENSOR)
    if PRETRAINED_MODEL:
        model = get_trained_model(weight_path=TRAINED_WEIGHT_PATH)
    else:
        model = net

    model = model.to(DEVICE)
    out_dataloaders, in_dataloaders = get_dataloaders()
    train_other_head(model, out_dataloaders, in_dataloaders)
