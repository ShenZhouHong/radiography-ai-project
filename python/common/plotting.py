#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import keras
import tensorflow_addons as tfa
import pickle

def plot_history(model_history, model_num: int=1, plot_acc: bool=True, plot_loss: bool=False):
    """
    Plot the training and validation metrics (AUC and loss) for a given model's history.

    Args:
        model_history (keras.callbacks.History): A Keras callback containing the model's training and validation history.
        model_num (int): An integer representing the number of the model (default: 1).
        plot_acc (bool): A boolean representing whether to plot the training and validation AUC (default: True).
        plot_loss (bool): A boolean representing whether to plot the training and validation loss (default: False).
    Returns:
        None: This function plots the metrics but does not return any value.
    """
    metric_name = [k for k in model_history.history.keys() if k.startswith('auc')][0]
    metric = model_history.history[metric_name]
    val_metric = model_history.history[f'val_{metric_name}']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(metric) + 1)

    if plot_acc:
        plt.plot(epochs, metric, 'r', label='Training AUC')
        plt.plot(epochs, val_metric, 'b', label='Validation AUC')

        plt.title(f'Model {str(model_num)}: Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend(loc='lower right')

    if plot_loss:
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        
        plt.title(f'Model {str(model_num)}: Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')

    plt.show()

def plot_kfold_history(k_fold_metrics: list[dict[str, float]], x_axis_max: int = 50, title: str = 'K-Fold Validation Metrics (K = 10)') -> None:
    """
    Plot the average and per-fold training and validation metrics (AUC) for a K-fold cross validation.

    Args:
        k_fold_metrics (list[dict[str, float]]): A list of dictionaries, each containing the metrics for one fold of the K-fold cross validation.

    Returns:
        None: This function plots the metrics but does not return any value.
    """
    avg_metrics: dict[str, list[float]] = calculate_mean_metrics(k_fold_metrics)

    fig = plt.figure(figsize=(10,10))

    # Individual Plots
    for fold in k_fold_metrics:
        metric_name = [k for k in avg_metrics.keys() if k.startswith('auc')][0]
        metric = fold[metric_name]
        val_metric = fold[f'val_{metric_name}']
        loss = fold['loss']
        val_loss = avg_metrics['val_loss']
        epochs = range(1, len(metric) + 1)

        plt.plot(epochs, metric, c='r', linewidth=1, alpha=0.1, label="_Per Fold Training AUC")
        plt.plot(epochs, val_metric, c='b', linewidth=1, alpha=0.25, label="_Per Fold Validation Auc")


    # The K-Fold Average
    metric_name = [k for k in avg_metrics.keys() if k.startswith('auc')][0]
    metric = avg_metrics[metric_name]
    val_metric = avg_metrics[f'val_{metric_name}']
    loss = avg_metrics['loss']
    val_loss = avg_metrics['val_loss']
    epochs = range(1, len(metric) + 1)

    plt.plot(epochs, metric, 'r', label='Avg. Training AUC')
    plt.plot(epochs, val_metric, 'b', label='Avg. Validation AUC')

    # Plot line of max val_auc
    plt.axhline(y=max(val_metric), c='b', label='Highest Avg. Validation AUC', linestyle="dotted")
    plt.text(x_axis_max - 10, max(val_metric) + 0.005,f"y = {round(max(val_metric), 3)}", c="b")

    # Now set title, ticks, and labels
    plt.title(title)
    plt.xlabel('Epochs')
    plt.xticks(np.arange(1, x_axis_max, 5))
    plt.ylabel('Avg. AUC')
    plt.yticks(np.arange(0.5, 1, 0.05))
    plt.legend(loc='lower right')

    plt.show()

    return None


def preview_dataset(
        ds: tf.data.Dataset,
        size: tuple[int, int] = (12, 12),
        grid: int = 3
    ) -> plt.Figure:
    """
    Preview a TensorFlow dataset by plotting a grid of images with their corresponding labels.

    Args:
        ds (tf.data.Dataset): A TensorFlow dataset containing images and labels.
        size (tuple[int, int]): A tuple representing the size of the resulting figure (default: (12, 12)).
        grid (int): An integer representing the number of rows and columns in the resulting grid (default: 3).
    Returns:
        plt.Figure: A Matplotlib figure object containing the plotted images.
    Raises:
        ValueError: If the input dataset is not in a valid format (rank 3 or 4).
    """
    
    fig, axs = plt.subplots(grid, grid, figsize=size)

    # Cache the dataset before calling take
    ds = ds.cache()
    
    for i, (images, labels) in enumerate(ds.take(grid * grid)):
        row: int = i // 3
        col: int = i %  3

        # rank 4: (batch_size, height, width, channels)
        # rank 3: (height, width, channels)
        ds_rank: int = len(images.shape)

        if ds_rank == 4:
            # If the dataset is batched, we need to select for a sample out of the batch.
            image: tf.Tensor = images[0]
            label: tf.Tensor = labels[0]
        elif len(images.shape) == 3:
            # Otherwise if the image is not batched, we simply graph it directly.
            image: tf.Tensor = images
            label: tf.Tensor = labels
        else:
            raise ValueError

        axs[row][col].imshow((image+1)/2)
        axs[row][col].set_title(str(label.numpy()))
        axs[row][col].axis('off')
        
        plt.close()
        
    return fig
