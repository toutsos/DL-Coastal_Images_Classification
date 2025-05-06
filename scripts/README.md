# Project Script Descriptions

## 1. `COASTresnet50.sh`

This script trains a Resnet50 model for coastal image classification. You can change most of the hyperparameters through this script, and if a hyperparam doesn't exist right now, it can be explicity added.

- For **Fine-tune** a model use the flag: `module.fine_tune=true` else it will trained with **Fixed-features**.
- For loading an already existing model and train it from the scratch: use the flag: `module.use_saved_model=true` and paste the path for the `checkpoint` in the config file.
