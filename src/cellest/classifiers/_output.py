import re
import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cellest.utils import generate_random_strings

def create_output_dir(config):
    source_name = config.model.model_nickname
    if source_name is None:
        source_name = config.model.arch
        if config.model.premodel is not None:
            source_name += f"_{config.model.premodel}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = f"{source_name}_{timestamp}_{generate_random_strings()}"
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name).strip("_.-")
    run_name = run_name or "cellest_run"
    output_dir = config.train.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return source_name, run_name, output_dir

def save_run_outputs(output_dir, history, curves, model_metadata, summary):
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame({"fpr": curves["fpr"], "tpr": curves["tpr"]}).to_csv(
        output_dir / "roc.csv", index=False
    )
    pd.DataFrame(
        {"precision": curves["precision"], "recall": curves["recall"]}
    ).to_csv(output_dir / "pr.csv", index=False)
    np.save(output_dir / "confmat.npy", curves["confusion_matrix"])
    torch.save(model_metadata, output_dir / "model.pt")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def save_run_plots(history, curves, class_names, run_name, evaluation_label, output_dir):
    history_df = pd.DataFrame(history)

    fig, ax = plt.subplots()
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val loss")
    ax.set(xlabel="epoch", ylabel="loss", title=f"{run_name} -- loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(history_df["epoch"], history_df["train_acc"], label="train acc")
    ax.plot(history_df["epoch"], history_df["val_acc"], label="val acc")
    ax.set(xlabel="epoch", ylabel="accuracy", title=f"{run_name} -- accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "acc_curve.png", dpi=150)
    plt.close(fig)

    if "val_auc" in history_df:
        fig, ax = plt.subplots()
        ax.plot(history_df["epoch"], history_df["val_auc"], label="val AUROC")
        ax.set(xlabel="epoch", ylabel="AUROC", title=f"{run_name} -- val AUROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "auroc_curve.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(
        curves["fpr"],
        curves["tpr"],
        label=f"{evaluation_label} AUC = {curves['roc_auc']:.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{run_name} -- {evaluation_label} ROC",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(curves["recall"], curves["precision"])
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{run_name} -- {evaluation_label} PR",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "pr.png", dpi=150)
    plt.close(fig)

    confusion = curves["confusion_matrix"]
    fig, ax = plt.subplots()
    image = ax.imshow(confusion, cmap="Blues")
    if len(class_names) == 2:
        display_names = class_names
    else:
        display_names = [f"not {class_names[0]}", class_names[0]]
    ax.set_xticks([0, 1], labels=display_names)
    ax.set_yticks([0, 1], labels=display_names)
    ax.set(
        xlabel="Predicted",
        ylabel="True",
        title=f"{run_name} -- {evaluation_label} confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            color = "white" if confusion[row, column] > confusion.max() / 2 else "black"
            ax.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
                color=color,
            )
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(output_dir / "confmat.png", dpi=150)
    plt.close(fig)

