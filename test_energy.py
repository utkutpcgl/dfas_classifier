import torch
import numpy as np
from torchvision import datasets, transforms

from display_ood_test_results import show_performance, get_measures, print_measures, print_and_log
from train_energy import SEED
from classifier_net import net
from dataloader import DATA_DIR, DEVICE, data_transforms, HYPS, image_datasets


def get_trained_network(model, weight_path):
    """Get the network based on the specifications.
    Args:
        weight_path (str): Path of the weight of the target model.
    """
    model.load_state_dict(torch.load(weight_path))
    return model


def get_val_dataloaders():
    data_dir = DATA_DIR / "OOD"
    out_image_dataset_val = datasets.ImageFolder(data_dir / "val", data_transforms["val"])
    # pin_memory=True to speed up host to device data transfer with page-locked memory
    out_dataloader_val = torch.utils.data.DataLoader(
        out_image_dataset_val,
        batch_size=HYPS["energy_batch_size_OD"],
        shuffle=True,
        num_workers=HYPS["workers"],
        pin_memory=True,
    )
    in_dataloader_val = torch.utils.data.DataLoader(
        image_datasets["val"],
        batch_size=HYPS["energy_batch_size_TEST"],
        shuffle=True,
        num_workers=HYPS["workers"],
        pin_memory=True,
    )
    return out_dataloader_val, in_dataloader_val


# CALCULATE AND DISPLAY SCORES
def get_ood_scores(
    loader,
    ResNet,
    in_dist=False,
    # batch_size_test=HYPS["energy_batch_size_TEST"],
    energy_temperature=HYPS["energy_temperature"],
):
    """Get energy scores would be a better name as the method is also used for correct classifications."""
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    _score = []  # The ID data energy scores are stored here.
    # The scores below are used to find the error rate at the original in distribution task.
    # Number of right/ predictions are used rather than their sofmtax score.
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # Do not reduce the OOD samples by commenting below break code.:
            # if batch_idx >= ood_num_examples // batch_size_test and in_dist is False:
            #     break

            data = data.to(DEVICE)

            output = ResNet(data)
            smax = to_np(torch.nn.functional.softmax(output, dim=1))

            _score.append(-to_np((energy_temperature * torch.logsumexp(output / energy_temperature, dim=1))))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                # Calculate the scores, only the lenght of those score lists are used.
                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()


def get_and_print_results(ood_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, log_path):
    aurocs, auprs, fprs = [], [], []
    out_score = get_ood_scores(ood_loader, ResNet)
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])
    print_and_log(in_score[:3], out_score[:3], log_path)
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    print_measures(auroc, aupr, fpr, log_path=log_path)


def show_right_wrong_perf(right_score, wrong_score, log_path):
    num_right = len(right_score)
    num_wrong = len(wrong_score)
    print_and_log("Error Rate {:.2f}".format(100 * num_wrong / (num_wrong + num_right)), log_path)
    # /////////////// Error Detection ///////////////
    print_and_log("\n\nError Detection", log_path)
    show_performance(wrong_score, right_score, log_path)


def get_ood_results(ood_val_loader, in_score, ResNet, log_path, batch_size_test=HYPS["energy_batch_size_TEST"]):
    auroc_list, aupr_list, fpr_list = [], [], []
    print_and_log("\n\Other (OOD) Detection", log_path)
    get_and_print_results(ood_val_loader, in_score, auroc_list, aupr_list, fpr_list, ResNet, log_path=log_path)
    print_and_log("\n\nMean Test Results!!!!!", log_path)
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), log_path=log_path)


def test_network(ResNet, log_path):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True  # fire on all cylinders
    out_dataloader_val, in_dataloader_val = get_val_dataloaders()
    # Only in_score is important below.print_and_log
    in_score, right_score, wrong_score = get_ood_scores(in_dataloader_val, ResNet, in_dist=True)
    show_right_wrong_perf(right_score, wrong_score, log_path)
    get_ood_results(out_dataloader_val, in_score, ResNet, log_path=log_path)
