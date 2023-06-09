# ERA-S6

# Handwritten Digits Classification using Convolution Neural Network (CNN)

This repo contains code for building a CNN model which is trained on the MNIST dataset for Handwritten Digits Classification

## File Structure in S6 Folder

`model.py` contains the CNN Model. Summary of the Model - 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
              ReLU-2           [-1, 32, 26, 26]               0
       BatchNorm2d-3           [-1, 32, 26, 26]              64
         Dropout2d-4           [-1, 32, 26, 26]               0
         MaxPool2d-5           [-1, 32, 13, 13]               0
            Conv2d-6           [-1, 32, 13, 13]           1,024
              ReLU-7           [-1, 32, 13, 13]               0
            Conv2d-8           [-1, 32, 11, 11]           9,216
              ReLU-9           [-1, 32, 11, 11]               0
      BatchNorm2d-10           [-1, 32, 11, 11]              64
        Dropout2d-11           [-1, 32, 11, 11]               0
        MaxPool2d-12             [-1, 32, 5, 5]               0
           Linear-13                   [-1, 10]           8,000
================================================================
Total params: 18,656
Trainable params: 18,656
Non-trainable params: 0
----------------------------------------------------------------
```

`utils.py` contains the utility functions for training the model, testing the model

`S6.ipynb` contains the code for Using the MNIST Dataset and training the model

## How to Use

Open the `S6.ipynb` file and Model Can be trained using the MNIST Data. Alternatiely `model.py` file can be imported directly in any another notebook and can be trained on other datasets and saved to the disk.

Standalone, `utils.py` can also be used for the generic train, test irrespective of the notebook file and the model file.