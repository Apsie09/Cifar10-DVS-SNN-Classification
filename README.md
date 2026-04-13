# CIFAR10-DVS Classification with Spiking Neural Networks

This repository contains a university course project on event-based image classification using `PyTorch`, `snnTorch`, and `Tonic` on the `CIFAR10-DVS` dataset.

The project focuses on building a clean and reproducible pipeline for:
- loading and inspecting event streams
- converting events to frame-based temporal tensors
- training hybrid CNN + SNN classifiers
- evaluating architecture and preprocessing choices

## Project Goal

The goal of the project is to study how spiking neural network methods can be applied to event-based vision data and to evaluate a practical classifier for `CIFAR10-DVS`.

The final selected model in this project is a wider hybrid `CNN + SNN` architecture trained on a `10-bin` temporal representation of the input events.

## Dataset

The project uses the `CIFAR10-DVS` dataset introduced in:

Hongmin Li, Hanchao Liu, Xiangyang Ji, Guoqi Li, and Luping Shi,  
`CIFAR10-DVS: An Event-Stream Dataset for Object Classification`,  
*Frontiers in Neuroscience*, 2017.

The dataset is not included in this repository. It should be downloaded locally and stored under `data/raw/`.

In this project, dataset access is handled through `Tonic`.

## Best Result

The strongest result obtained in the project is:

- model: `wider_10bins_10epochs`
- representation: `10` temporal bins, `2` polarity channels, normalization enabled
- test accuracy: `52.93%`

This model outperformed:
- the smaller baseline model
- a stronger weight-decay variant
- a longer 15-epoch run
- a more SNN-heavy Nengo-inspired architecture

## Repository Structure

```text
.
├── environment.yml
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_preprocessing_experiments.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation_and_results.ipynb
│   └── 05_experiment_comparison.ipynb
├── outputs/
│   └── figures/
├── src/
│   └── snn_cifar10dvs/
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── models.py
│       ├── preprocessing.py
│       ├── train.py
│       └── utils.py
└── README.md
```

## Environment Setup

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate snn-cifar10dvs
```

Register it as a Jupyter kernel:

```bash
python -m ipykernel install --user --name snn-cifar10dvs --display-name "Python (snn-cifar10dvs)"
```

## Project Workflow

The project was developed in stages:

1. dataset inspection and validation of the CIFAR10-DVS loading path
2. preprocessing comparison for multiple temporal bin counts
3. baseline model training
4. evaluation and result interpretation
5. controlled architecture comparison

The notebooks are intended to be read in that order.

## Included Figures

The repository includes selected figures generated during the project:

- dataset sample frames
- preprocessing comparison
- experiment comparison
- final confusion matrix

These are stored in `outputs/figures/`.
