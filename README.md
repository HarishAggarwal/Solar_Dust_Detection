# CNN for Solar Dust Detection

This repository contains a Convolutional Neural Network (CNN) implemented using TensorFlow and Keras for detecting dust on solar panels. The model is trained on an image dataset using data augmentation techniques to improve its generalization ability.

## Project Overview

The goal of this project is to classify images into two categories: clean and dusty solar panels. The model is built using a series of convolutional layers to extract features from the images and a fully connected network to make predictions.

### Features

- **Data Augmentation**: Applied transformations such as rotation, shifting, zooming, and flipping to artificially expand the dataset.
- **Binary Classification**: The model classifies images into two categories (clean/dusty).
- **Training and Validation Splits**: 80% of the data is used for training, and 20% is used for validation, all split from the same directory.

## Data Augmentation and Preprocessing

We use `ImageDataGenerator` to load the image data, applying the following augmentation techniques:
- **Rotation**: Randomly rotate the images by up to 40 degrees.
- **Width/Height Shifting**: Randomly shift the images along both axes by 20%.
- **Shear**: Apply random shearing transformations.
- **Zooming**: Randomly zoom in on the images by up to 20%.
- **Horizontal Flipping**: Flip the images horizontally.

The image pixel values are rescaled to the range `[0, 1]` by dividing by 255.

## Model Architecture

The CNN model is composed of three convolutional layers followed by max-pooling layers. The architecture is designed to extract spatial features from the images. The network ends with a fully connected layer and a sigmoid activation function to output a probability for binary classification.

**Model Summary**:
- **Input Shape**: `(224, 224, 3)` (RGB images)
- **Convolutional Layers**:
  - Conv2D (32 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
  - Conv2D (128 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
- **Fully Connected Layer**: 512 units + ReLU
- **Output Layer**: 1 unit + Sigmoid (for binary classification)
