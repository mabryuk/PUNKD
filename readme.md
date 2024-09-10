# Overview

This is a collection of notebooks and scripts that goes through the process of distilling a U-Net64 model (teacher) into a U-Net16 model (student). Both models are trained on the [BRATS 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html) using pytorch. The dataset is preprocessed then both the teacher and the student model is trained for 100 epochs. The student model is then evaluated and compared against the teacher model and a model that is of the same size but lacking knowledge distillation.

# Installing dependencies

[CUDA](https://developer.nvidia.com/cuda-downloads) Version 12.3 >= is required to run the code.

[Python](https://www.python.org/downloads/) Version 3.10.x >= is required to run the code.

# Setting up a virtual environment

## Create a virtual environment
```bash
python3 -m venv env
```

## Activate the virtual environment

```bash
source env/bin/activate
```

## Install modules

```bash
python3 -m pip install -r requirements.txt
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


