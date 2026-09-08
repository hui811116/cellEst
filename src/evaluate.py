import torch
from torcheval.metrics.functional import binary_auprc, binary_accuracy, binary_auroc, binary_f1_score, binary_recall

def metrics_all(y_pred,y_true):
    y_hard = y_pred.argmax(dim=1)
    y_prob = y_pred[:,1]
    bin_acc = binary_accuracy(y_hard,y_true)
    bin_prc = binary_auprc(y_prob,y_true)
    bin_roc = binary_auroc(y_prob,y_true)
    bin_f1 = binary_f1_score(y_hard,y_true)
    bin_recall = binary_recall(y_hard,y_true)
    return {"acc":bin_acc.item(),
            "prc":bin_prc.item(),
            "roc":bin_roc.item(),
            "f1":bin_f1.item(),
            "recall":bin_recall.item()}