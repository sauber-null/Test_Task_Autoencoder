# Denoising Autoencoder

The main goal of the Denoising Autoencoder is to process input image with the graph and handwritten text and return denoised gray scaled image.

## Installation

1. The project can be cloned using the SSH:

```bash
git clone git@gitlab.mobidev.biz:d.zahoruiko/test-task-autoencoder.git
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Download and extract the data:
- Download dataset with the [link](https://drive.google.com/file/d/1QQuXiggreSYWXiSruuKwWgdJlUZnkwTm/view?usp=sharing)
- Create paths "<project_folder>/data/AUTOENCODER/augmented_src" and "<project_folder>/data/AUTOENCODER/augmented_labels"
- Unzip the content of the corresponding sub-folders in the zip archive with the dataset to a newly created folders

## Usage

### Training

By default, the model will train with such a set up:

- Model: custom model with 4 Conv2d-BatchNorm-ReLU blocks in encoder and learnable decoder
- Loss: BCEWithLogits (without weights)
- Optimizer: Adam (learning rate = 0.0001, betas: 0.7, 0.999)
- Device: Cuda
- Epochs: 5
- Image size: 640x640
- Batch size: 8

Run this command in the project folder:
```bash
python3 main.py --mode='train'
```

You will be able to see the graphics of the training run entering this command in the project folder:
```bash
tensorboard --logdir=runs
```

### Inference

To run the inference, enter this command in the project folder:
```bash
python3 main.py --mode='inference'
```

## Report
You can find the report on this task [here](https://docs.google.com/document/d/1LZ24Mde_-Hn6CfufTX0UKTsfr6yJso1oqgCkqLynhlQ/edit?usp=sharing)
