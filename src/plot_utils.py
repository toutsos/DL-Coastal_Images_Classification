import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(log_dir: str):
    metrics_csv = f"{log_dir}/metrics.csv"
    metrics_df = pd.read_csv(metrics_csv)

    if "loss" in metrics_df.columns and "val_loss" in metrics_df.columns:
        trn_loss_df = metrics_df[["epoch", "loss"]].dropna()
        val_loss_df = metrics_df[["epoch", "val_loss"]].dropna()

        plt.figure()
        plt.plot(trn_loss_df["epoch"], trn_loss_df["loss"], label="Training")
        plt.plot(val_loss_df["epoch"], val_loss_df["val_loss"], label="Validation")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{log_dir}/loss.png")
        plt.close()

    if "acc" in metrics_df.columns and "val_acc" in metrics_df.columns:
        trn_acc_df = metrics_df[["epoch", "acc"]].dropna()
        val_acc_df = metrics_df[["epoch", "val_acc"]].dropna()

        plt.figure()
        plt.plot(trn_acc_df["epoch"], trn_acc_df["acc"], label="Training")
        plt.plot(val_acc_df["epoch"], val_acc_df["val_acc"], label="Validation")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{log_dir}/accuracy.png")
        plt.close()


def plot_test_metrics(log_dir: str):
    metrics_csv = f"{log_dir}/metrics.csv"
    metrics_df = pd.read_csv(metrics_csv)

    print("Loaded metrics columns:", metrics_df.columns)
    print(metrics_df.tail())  # Show the last few rows to verify test values

    if "test_loss" in metrics_df.columns and metrics_df["test_loss"].dropna().any():
        test_loss_df = metrics_df[["epoch", "test_loss"]].dropna()

        plt.figure()
        plt.plot(test_loss_df["epoch"], test_loss_df["test_loss"], label="Test Loss", color="red")
        plt.title("Test Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{log_dir}/test_loss.png")
        plt.close()

    if "test_acc" in metrics_df.columns and metrics_df["test_acc"].dropna().any():
        test_acc_df = metrics_df[["epoch", "test_acc"]].dropna()

        plt.figure()
        plt.plot(test_acc_df["epoch"], test_acc_df["test_acc"], label="Test Accuracy", color="blue")
        plt.title("Test Accuracy Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{log_dir}/test_accuracy.png")
        plt.close()
