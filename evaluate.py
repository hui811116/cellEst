import torch
from torcheval.metrics.functional import binary_auprc, binary_accuracy, binary_auroc, binary_f1_score, binary_recall
from torcheval.metrics.functional import multiclass_auprc, multiclass_accuracy, multiclass_auroc, multiclass_f1_score, multiclass_recall


def metrics_all(y_pred,y_true):
    """Evaluation metrics"""
    y_hard = y_pred.argmax(dim=1)
    y_prob = y_pred[:,1]
    #bin_acc = binary_accuracy(y_hard,y_true)
    acc = multiclass_accuracy(y_hard,y_true)
    #bin_prc = binary_auprc(y_prob,y_true)
    prc = multiclass_auprc(y_prob,y_true)
    #bin_rocbin_roc = binary_auroc(y_prob,y_true)
    roc = multiclass_auroc(y_prob,y_true)
    #bin_f1 = binary_f1_score(y_hard,y_true)
    f1 = multiclass_f1_score(y_hard,y_true)
    #bin_recall = binary_recall(y_hard,y_true)
    recall = multiclass_recall(y_hard,y_true)
    return {"acc":acc.item(),
            "prc":prc.item(),
            "roc":roc.item(),
            "f1":f1.item(),
            "recall":recall.item()}