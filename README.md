# DevaNet
A lightweight Convolutional Neural Network for accurate classification of Devanagari digits (०–९)

## Overview
DevaNet is a Convolutional Neural Network (CNN) designed to classify handwritten Devanagari digits (०–९) with high accuracy.

The project focuses on two key aspects:

- Model performance:  optimizing a lightweight CNN architecture to achieve strong classification accuracy on handwritten digit datasets
- User interaction:  providing a simple GUI that allows users to draw digits and receive real-time predictions

This makes DevaNet both a machine learning experiment and an interactive application demonstrating practical deployment of a trained model.

## Requirements
- uv ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))

## Getting Started
```bash
git clone https://github.com/OmericoN/DevaNet.git
cd DevaNet
uv sync
```

## The Dataset
> The dataset consists of handwritten Devanagari digits (०–९) formatted as 32×32 RGB images.

#### Structure
- **Training set:** 34,000 labeled images
- **Test set:** 3,000 unlabeled images (used for Kaggle submission/evaluation)

#### Image Details
- Resolution: 32 × 32 pixels
- Channels: 3 (RGB)
- Classes: 10 (digits ०–९)
