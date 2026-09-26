from torcheval.metrics.functional import multiclass_auprc, multiclass_accuracy, multiclass_auroc, multiclass_f1_score, multiclass_recall

def metrics_all(y_pred,y_true):
    """Evaluation metrics"""
    y_hard = y_pred.argmax(dim=1)
    y_prob = y_pred[:,1]
    acc = multiclass_accuracy(y_hard,y_true)
    prc = multiclass_auprc(y_prob,y_true)
    roc = multiclass_auroc(y_prob,y_true)
    f1 = multiclass_f1_score(y_hard,y_true)
    recall = multiclass_recall(y_hard,y_true)
    return {"acc":acc.item(),
            "prc":prc.item(),
            "roc":roc.item(),
            "f1":f1.item(),
            "recall":recall.item()}