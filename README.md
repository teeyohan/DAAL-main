# DAAL-main

PyTorch implementation of "DAAL: Discrepancy-Aware Active Learning for Endoscopic Image Segmentation via Human-Machine Chromatic Discrepancy".

<div align="center">
  <img width="80%" alt="" src="DAAL.png">
</div>

## Setup
The following dependencies are recommended for the installation of the environment.

- python 3.8.13
- torch 1.11.0
- torchvision 0.12.0

## Dataset
It is recommended to download the CVC-ClinicDB dataset from the official website:

- CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/

, and place it in the "DAAL-main/data/" directory.

## Training and evaluation
For training and evaluation, use the following script:

- `python main.py`

, where config.py is the configuration file.
