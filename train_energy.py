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

from classifier_net import net
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
ENERGY_TRAINED_WEIGHT_PATH = Path(
    "/home/utku/Documents/repos/dfas_classifier/trained_models/exp_light_dataset_v1_classifier_resnet18_1/light_dataset_v1_classifier_resnet18_f1_energy.pt"
)
# device = torch.device("cuda:0" if torch)
SEED = 1
GET_ENERGY = False


def get_trained_model(weight_path):
    net.load_state_dict(torch.load(weight_path))
    return net


def get_dataloaders():
    data_dir = DATA_DIR / "OOD"
    out_image_datasets = {
        d_type: datasets.ImageFolder(data_dir / d_type, data_transforms[d_type]) for d_type in ["train", "val"]
    }
    # pin_memory=True to speed up host to device data transfer with page-locked memory
    out_dataloaders = {
        d_type: torch.utils.data.DataLoader(
            out_image_datasets[d_type],
            batch_size=HYPS["energy_batch_size_OD"],
            shuffle=True,
            num_workers=HYPS["workers"],
            pin_memory=True,
        )
        for d_type in ["train", "val"]
    }
    in_dataloaders = {
        d_type: torch.utils.data.DataLoader(
            image_datasets[d_type],
            batch_size=HYPS["energy_batch_size_ID"],
            shuffle=True,
            num_workers=HYPS["workers"],
            pin_memory=True,
        )
        for d_type in ["train", "val"]
    }
    return out_dataloaders, in_dataloaders


def get_avg_energy(dataloader, network):
    total_energy = 0
    total_num_samples = 0
    len_of_iterations = len(dataloader)
    for (data, label) in tqdm(dataloader, total=len_of_iterations):
        data = data.to(DEVICE)
        output = network(data)
        Energy_in = -sum(torch.logsumexp(output, dim=1)).cpu().item()
        total_energy += Energy_in
        total_num_samples += len(data)
    avg_energy = total_energy / total_num_samples
    return avg_energy


def get_average_energy_of_dataset(out_dataloader, in_dataloader, network):
    """Get the average energy to be used as m_in or m_out for the loss."""
    averge_energy_out = get_avg_energy(out_dataloader, network)
    averge_energy_in = get_avg_energy(in_dataloader, network)
    return averge_energy_out, averge_energy_in


def train_energy_one_epoch(
    model, dataloader_train_in, dataloader_train_out, scheduler, optimizer, lambda_energy, m_in, m_out, state_report
):
    model.train()  # enter train mode
    loss_avg = 0.0
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    total_tqdms = len(dataloader_train_out)
    for in_set, out_set in tqdm(zip(dataloader_train_in, dataloader_train_out), total=total_tqdms):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        data, target = data.to(DEVICE), target.to(DEVICE)
        # forward
        x = model(data)
        # backward
        optimizer.zero_grad()
        # loss = torch.nn.functional.cross_entropy(x[: len(in_set[0])], target)
        loss_classification = torch.nn.functional.cross_entropy(x[: len(in_set[0])], target)

        # cross-entropy from softmax distribution to uniform distributio
        Ec_out = -torch.logsumexp(x[len(in_set[0]) :], dim=1)
        Ec_in = -torch.logsumexp(x[: len(in_set[0])], dim=1)
        loss_ood = (
            torch.pow(torch.nn.functional.relu(Ec_in - m_in), 2).mean()
            + torch.pow(torch.nn.functional.relu(m_out - Ec_out), 2).mean()
        )
        # loss += lambda_energy * (
        #     torch.pow(torch.nn.functional.relu(Ec_in - m_in), 2).mean()
        #     + torch.pow(torch.nn.functional.relu(m_out - Ec_out), 2).mean()
        # )
        # number_of_tasks = 2
        task_normalized_weights = torch.nn.functional.softmax(torch.randn(2), dim=-1)
        loss = task_normalized_weights[0]*loss_classification + task_normalized_weights[1]*loss_ood
        loss.backward()
        optimizer.step()
        scheduler.step()  # update the steps of cosine annealing every step (over batches.)
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state_report["avg_train_loss"] = loss_avg


def validate_energy_one_epoch(model, val_loader, state_report):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # forward
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)  # Default is mean reduction.
            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            # test loss average
            total_loss += float(loss.data)

    state_report["in_avg_val_loss"] = total_loss / len(val_loader)
    state_report["in_avg_val_accuracy"] = correct / len(val_loader.dataset)


def train_energy_ft(
    model,
    dataloader_train_in,
    dataloader_train_out,
    dataloader_val_in,
    optimizer,
    scheduler,
    epochs,
    weight_path: Path,
    log_path,
    lambda_energy,
    m_in,
    m_out,
):
    """Fine tune an already trained model to detect out-of-dist samples (other class)."""
    print("Start energy fine tuning")
    state_report = {}
    for epoch in range(epochs):
        state_report["epoch"] = epoch
        begin_epoch = time.time()
        train_energy_one_epoch(
            model,
            dataloader_train_in=dataloader_train_in,
            dataloader_train_out=dataloader_train_out,
            scheduler=scheduler,
            optimizer=optimizer,
            lambda_energy=lambda_energy,
            m_in=m_in,
            m_out=m_out,
            state_report=state_report,
        )
        validate_energy_one_epoch(model=model, val_loader=dataloader_val_in, state_report=state_report)
        # Save model
        torch.save(model.state_dict(), weight_path.with_name(weight_path.stem + "_energy.pt"))
        state_report_str = "Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | In Avg. Val Loss {3:.3f} | In Avg. Val Accuracy {4:.2f}".format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state_report["avg_train_loss"],
            state_report["in_avg_val_loss"],
            100.0 * state_report["in_avg_val_accuracy"],
        )
        write_log(info_str=state_report_str, path=log_path)
        print(state_report)


def train_energy(model, out_dataloaders, in_dataloaders):
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYPS["energy_initial_lr"])
    if len(out_dataloaders["train"]) > len(in_dataloaders["train"]):
        total_num_steps = len(in_dataloaders["train"])
    else:
        total_num_steps = len(out_dataloaders["train"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=total_num_steps, eta_min=HYPS["energy_final_lr"]
    )
    log_path = TRAINED_WEIGHT_PATH.with_name(TRAINED_WEIGHT_PATH.stem + "_energy_train_log.txt")
    train_energy_ft(
        model=model,
        dataloader_train_in=in_dataloaders["train"],
        dataloader_train_out=out_dataloaders["train"],
        dataloader_val_in=in_dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=HYPS["energy_epochs"],
        weight_path=TRAINED_WEIGHT_PATH,
        log_path=log_path,
        lambda_energy=HYPS["energy_lambda"],
        m_in=HYPS["energy_m_in"],
        m_out=HYPS["energy_m_out"],
    )


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(HYPS["CUDA_FRACTION"], device=DEVICE)
    if GET_ENERGY:
        model = get_trained_model(weight_path=ENERGY_TRAINED_WEIGHT_PATH)
    else:
        model = get_trained_model(weight_path=TRAINED_WEIGHT_PATH)

    model = model.to(DEVICE)
    out_dataloaders, in_dataloaders = get_dataloaders()
    if GET_ENERGY:
        averge_energy_out, averge_energy_in = get_average_energy_of_dataset(
            out_dataloaders["train"], in_dataloaders["train"], model
        )
        print(f"averge_energy_out, averge_energy_in : {averge_energy_out}, {averge_energy_in}")
    else:
        train_energy(model, out_dataloaders, in_dataloaders)
