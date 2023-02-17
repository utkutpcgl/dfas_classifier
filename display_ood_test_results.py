import numpy as np
import sklearn.metrics as sk
from dataloader import HYPS
from copy import copy

RECALL_LEVEL_DEFAULT = 0.85
ENERGY_THRESHOLD = (HYPS["energy_m_in"] + HYPS["energy_m_out"]) / 2


def print_and_log(message, log_path: str):
    print(message)
    with open(log_path, "a") as message_appender:
        message_appender.write(message + "\n")


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("cumsum was found to be unstable: " "its last element does not correspond to sum")
    return out


def fpr_and_fdr_at_recall(y_true_np, y_score_np, recall_level=RECALL_LEVEL_DEFAULT, pos_label=None):
    """y_true are labels and labels[: len(pos)] = 1
    positives are -in_score, negatives are -out_score, where in_score and out_score are the original energy scores of samples!
    Strangely the use -score!!!!"""
    y_true = copy(y_true_np)
    y_score = copy(y_score_np)
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    # Thresholds are chosen dynamically to adjust the fpr at 95. (FPR (at TPR95%))
    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def accuracy_ood(y_true, y_score, energy_threshold=ENERGY_THRESHOLD):
    """y_true are labels and labels[: len(pos)] = 1
    positives are -in_score, negatives are -out_score, where in_score and out_score are the original energy scores of samples!
    Strangely the use -score!!!!"""
    y_score_energy_score = -y_score
    classes = np.unique(y_true)
    if not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    pos_label = 1

    # make y_true a boolean vector
    y_true = y_true == pos_label
    total_num_of_predictions = len(y_score_energy_score)
    in_predictions = y_score_energy_score <= energy_threshold
    correct_predictions = in_predictions == y_true

    accuracy = sum(correct_predictions) / total_num_of_predictions
    return accuracy


def precision_ood(y_true, y_score, energy_threshold=ENERGY_THRESHOLD):
    """y_true are labels and labels[: len(pos)] = 1
    positives are -in_score, negatives are -out_score, where in_score and out_score are the original energy scores of samples!
    Strangely the use -score!!!!"""
    y_score_energy_score = -y_score
    classes = np.unique(y_true)
    if not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    pos_label = 1

    # make y_true a boolean vector
    y_true = y_true == pos_label
    total_num_of_predictions = len(y_score_energy_score)
    in_predictions = y_score_energy_score <= energy_threshold
    correct_predictions = in_predictions == y_true

    accuracy = sum(correct_predictions) / total_num_of_predictions
    return accuracy


def get_measures(_pos, _neg, recall_level=RECALL_LEVEL_DEFAULT):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
    accuracy = accuracy_ood(labels, examples)

    return auroc, aupr, fpr, accuracy


def show_performance(pos, neg, method_name="Ours", recall_level=RECALL_LEVEL_DEFAULT, log_path="log"):
    """
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    """

    auroc, aupr, fpr, accuracy = get_measures(pos[:], neg[:], recall_level)

    print_and_log("\t\t\t" + method_name, log_path)
    print_and_log("FPR{:d}:\t\t\t{:.2f}".format(int(100 * recall_level), 100 * fpr), log_path)
    print_and_log("AUROC:\t\t\t{:.2f}".format(100 * auroc), log_path)
    print_and_log("AUPR:\t\t\t{:.2f}".format(100 * aupr), log_path)
    print_and_log("Accuracy:\t\t\t{:.2f}".format(100 * accuracy), log_path)
    # print_and_log('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def print_measures(auroc, aupr, fpr, accuracy, method_name="Ours", recall_level=RECALL_LEVEL_DEFAULT, log_path="log"):
    print_and_log("\t\t\t\t" + method_name, log_path)
    print_and_log("  FPR{:d} AUROC AUPR".format(int(100 * recall_level)), log_path)
    print_and_log("& {:.2f} & {:.2f} & {:.2f}".format(100 * fpr, 100 * auroc, 100 * aupr), log_path)
    print_and_log(f"Accuracy at energy_threshold = (m_in + m_out)/2 = {100 * accuracy}", log_path)

    # print_and_log('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    # print_and_log('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    # print_and_log('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
