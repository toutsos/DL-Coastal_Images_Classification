# CS4321 Midterm Project
 - Milanese Roberto
 - Singley Jarrad
 - Angelos Toutsios

### Description

- For each model we've created its own `datamodule`,`model`,`config` and `run.sh` file.
- We have our applied hyperparams for both our **Fixed** and **Finetuned** models in the `MidtermReport.docx`.
- For all of our models we used a main template, but each one of us implemented some utilities with a minor different way.
- Our results and graphs are in the `MidtermReport.docx`.
- A detailed description of our script (.sh) files exist in the [Script Descriptions](scripts/descriptions.md)


## Setup Environment

For more information on using Pixi on Hamming and connecting to a GPU node, see the [Wiki](https://gitlab.nps.edu/cs4321/lightning_starter/-/wikis/home)

1. After connecting to Hamming, install Pixi if not already installed:

    `curl -fsSL https://pixi.sh/install.sh | bash`


2. Start a session on Hamming with a GPU. For example:

    `salloc -p genai --gres=gpu:1`


3. While in the project directory, install the environment with the following:

    `pixi install`

   It is important to be on a node with GPU(s) allocated. Otherwise, the above will not work.


4. Exit the GPU session:

    `exit`


5. For sbatch scripts, add the following before the execution of your python script so that
the pixi environment is activated:

    `eval "$(pixi shell-hook -s bash)"`

## Project packages:
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

More detais about the versions of each package you can find in the [Packages-Versions](pixi.lock)

