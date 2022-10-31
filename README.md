# Defeating DeepFakes via Adversarial Visual Reconstruction

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2.0](https://img.shields.io/badge/pytorch-1.2.0-green.svg?style=plastic)


> **Defeating DeepFakes via Adversarial Visual Reconstruction** <br>
> Ziwen He, Wei Wang, Weinan Guan, Jing Dong, Tieniu Tan <br>
> *ACM Multimedia (MM) 2022*

[[Paper](https://doi.org/10.1145/3503161.3547923)]

**NOTE:** Currently, this repository is a simple PyTorch example of defeating StarGAN.

## Pre-trained Models

Please download the pre-trained models from the following links and save them to `models/pretrain/`

| Description | Generator | Encoder |
| :---------- | :-------- | :------ |
| Model trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. | [face_256x256_generator](https://drive.google.com/file/d/1SjWD4slw612z2cXa3-n38JwKZXqDUerG/view?usp=sharing)    | [face_256x256_encoder](https://drive.google.com/file/d/1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO/view?usp=sharing)
| [Perceptual Model](https://drive.google.com/file/d/1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y/view?usp=sharing)

For the StarGAN, to download a pre-trained model checkpoint, cd StarGAN and run the script below. The pre-trained model checkpoint will be downloaded and saved into `./stargan_celeba_128/models` directory.

```bash
$ bash download.sh pretrained-celeba-128x128
```

## Datasets

Follow [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) to download the dataset and specify the path in invert.py.

## Usage
<p align="justify"> Simply run the following command

```
  python invert.py
```
<p> 

## Acknowledgment
This repo is based on [In-Domain GAN Inversion](https://github.com/genforce/idinvert_pytorch) and [StarGAN](https://github.com/yunjey/StarGAN), thanks for their great work.

## BibTeX

```bibtex
@inproceedings{10.1145/3503161.3547923,
author = {He, Ziwen and Wang, Wei and Guan, Weinan and Dong, Jing and Tan, Tieniu},
title = {Defeating DeepFakes via Adversarial Visual Reconstruction},
year = {2022},
doi = {10.1145/3503161.3547923},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {2464–2472}
}
```
