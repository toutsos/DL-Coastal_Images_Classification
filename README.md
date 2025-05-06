# DL-Coastal_Images_Classification

**Custom Deep Learning Pipeline for Coastal Image Classification**

This project provides a streamlined deep learning framework for classifying coastal images using a ResNet50-based architecture. The pipeline is modular and fully configurable, making it easy to train, evaluate, and customize experiments.

## Key Features

- **ResNet50 Model:**
  A custom-implemented ResNet50 model for image classification tasks.

- **Flexible Data Handling:**
  Utilities for dataset preparation, normalization, and basic data checks.

- **Configurable Experiments:**
  All key parameters (dataset path, batch size, epochs, optimizer settings) are controlled via YAML configs.

- **Training & Evaluation Scripts:**
  Easy-to-run shell scripts for both training and evaluation.

- **Pixi Environment:**
  Ensures fully reproducible environments via `pixi.toml`.

---

## Installation

1️⃣ **Clone the repository:**

```bash
git clone https://github.com/toutsos/DL-Coastal_Images_Classification.git
cd DL-Coastal_Images_Classification
```

2️⃣ **Set up the Pixi environment:**

```bash
pixi install
```

---

## Training

Run the training script:

```bash
bash scripts/COASTresnet50.sh
```

This will:

- Load the dataset using the configuration in `configs/COASTresnet50.yaml`
- Train the ResNet50 model
- Save checkpoints and logs

---

## Evaluation

To evaluate the trained model:

```bash
bash scripts/evaluation.sh
```

---

## Modifying Hyperparameters

Edit the YAML config file in `configs/` (e.g., `COASTresnet50.yaml`) to change:

- Dataset path
- Batch size
- Number of epochs
- Learning rate & optimizer parameters
- Image size & normalization settings

Example snippet from the YAML:

```yaml
batch_size: 32
num_epochs: 50
learning_rate: 0.001
dataset_path: /path/to/your/dataset
```

---

## Metrics & Logging

During training and evaluation, you'll see:

- **Training & Validation Accuracy**
- **Loss Curves**
- **Confusion Matrix** and visualizations (via `plot_utils.py`)

Logs and outputs are saved to the specified directory in your config.

---

## 🗂️ Repository Structure

```
├── configs/
│   └── COASTresnet50.yaml          # Configuration file
├── scripts/
│   ├── COASTresnet50.sh            # Training script
│   └── evaluation.sh               # Evaluation script
├── src/
│   ├── train.py                    # Training entry point
│   ├── evaluation.py               # Evaluation logic
│   ├── datamodules/
│   │   └── dataset.py              # Dataset loader
│   └── modules/
│       └── ResNet50Model.py        # ResNet50 architecture
├── pixi.toml                       # Environment specification
├── README.md
└── .gitignore
```

---

## How to Extend

- **New Dataset:**
  Modify the dataset loader in `datamodules/dataset.py` to handle new formats.

- **Model Tweaks:**
  Customize or extend `modules/ResNet50Model.py` to experiment with new architectures.

- **New Configs:**
  Duplicate the YAML config and adjust as needed for new experiments.

## Project dependencies:

- pytorch
- lightning
- torchmetrics
- torchvision
- matplotlib
- pandas
- numpy
- pyrootutils
- hydra-core
- pytorch-cuda
- webdataset
- scikit-learn

More detais about the versions of each package you can find in the [Packages-Versions](pixi.toml)
