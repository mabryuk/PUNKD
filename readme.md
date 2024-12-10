# Overview

This is a collection of notebooks and scripts that goes through the process of distilling a U-Net64 model (teacher) into a U-Net16 model (student). Both models are trained on the [BRATS 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html) using pytorch. The dataset is preprocessed then both the teacher and the student model is trained for 100 epochs. The student model is then evaluated and compared against the teacher model and a model that is of the same size but lacking knowledge distillation.

# Dev Container

The repository contains a devcontainer configuration that can be used to run the code in a container. To use the devcontainer, you need to have [Docker](https://docs.docker.com/get-docker/) and [Visual Studio Code](https://code.visualstudio.com/) installed on your machine. as well as the [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension pack for Visual Studio Code.

# Installing dependencies

[CUDA](https://developer.nvidia.com/cuda-downloads) Version 12.3 >= is required to run the code.

[Python](https://www.python.org/downloads/) Version 3.10.x >= is required to run the code.

# Setting up a virtual environment

## Create a virtual environment
```bash
conda create -n <ENV_NAME> python=<PYTHON_VERSION>
```

## Activate the virtual environment

```bash
conda activate <ENV_NAME>
```

## Add the pytorch and conda-forge channels
    
```bash
conda config --add channels pytorch
conda config --add channels conda-forge
```

## Install modules

```bash
conda update -n <ENV_NAME> -f environment.yml
```

# Adding credentials for archiving models

The models are uploaded to a cloudflare R2 bucket. To upload the models, you need to add your cloudflare credentials to a `.env` file in the root directory.
The `.env` file should look like this:

```bash
R2_ENDPOINT='' # The endpoint of the R2 bucket
R2_TOKEN='' # The auth token of the R2 bucket
R2_KEY=''  # The access key of the R2 bucket
R2_SECRET='' # The secret key of the R2 bucket
R2_BUCKET='' # The name of the bucket
```


