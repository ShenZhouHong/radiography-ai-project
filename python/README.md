# Implementation of the RUST Radiography AI model

This directory contains the Python code responsible for the training, data collection, and implementation of the model. The code is mostly included in the form of Jupyter notebooks, as well as a few standalone files that are imported by them.

## Initial Evaluation

The implementation begins with an initial evaluation of four models in order to establish an adequate baseline. These four models are:

* [LeNet (1998)](https://en.wikipedia.org/wiki/LeNet), a classical 'shallow' CNN. This model is evaluated to form a minimal baseline for performance.
* [InceptionV3](https://en.wikipedia.org/wiki/Inceptionv3),**end-to-end trained** on the dataset. This second model is used to form a baseline for the technique of *transfer-learning*. If the transfer-learning approach performs worse than simply training InceptionV3 on our dataset, then the premise of the project must be revisited.
* [InceptionV3](https://en.wikipedia.org/wiki/Inceptionv3), trained on the [ImageNet](https://en.wikipedia.org/wiki/ImageNet) dataset. This is our transfer-learning model, *prior* to any hyperparameter tuning.
* [InceptionV3](https://en.wikipedia.org/wiki/Inceptionv3), trained on the [RadImageNet](https://www.radimagenet.com/) dataset. This is our transfer-learning model using the alternative base weights. Once again *prior* to any hyperparameter tuning.

The models for the initial evaluation are implemented within their respective Jupyter notebooks, located within the `initial-evaluation/` directory.

## Hyperparameter Search

The notebook files for the hyperparameter search procedure is located within the `hyperparam-search/` directory. Within the folder, there are two notebooks:

* `regime-1.ipynb`
* `regime-2.ipynb`

These correspond to the hyperparameter search regime I, and the hyperparameter search regime II as defined in the methodology and implementation.

## Analysis

These notebooks are used to analyse the hyperparameter search and model data.

## Final Model

The final model folder contains the notebook used to train the final model (with the best hyperparameters), the model weights, an inference example, as well as related analysis code and data CSV files.