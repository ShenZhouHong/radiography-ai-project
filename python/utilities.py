#!/usr/bin/env python3

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import tensorflow_addons as tfa
import pickle

AUTOTUNE = tf.data.AUTOTUNE

def get_single_image(ds: tf.data.Dataset) -> tf.Tensor:
    """
    Get a single image from a TensorFlow dataset and returns it as a tf.Tensor.

    Args:
        ds (tf.data.Dataset): A TensorFlow dataset containing images.
    Returns:
        tf.Tensor: A TensorFlow tensor representing a single image from the dataset.
    Raises:
        ValueError: If the input dataset is not in a valid format (rank 3 or 4).
    """

    for (images, labels) in ds.take(1):
        # rank 4: (batch_size, height, width, channels)
        # rank 3: (height, width, channels)
        ds_rank: int = len(images.shape)

        if ds_rank == 4:
            # If the dataset is batched, we need to select for a sample out of the batch.
            image: tf.Tensor = images[0]
        elif len(images.shape) == 3:
            # Otherwise if the image is not batched, we simply graph it directly.
            image: tf.Tensor = images
        else:
            raise ValueError
        
    return image

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

def configure_for_performance(ds: tf.data.Dataset, batch_size: int = 128) -> tf.data.Dataset:
  """
  Configure a TensorFlow dataset for optimal performance by applying caching, batching, and prefetching.

  Args:
    ds (tf.data.Dataset): A TensorFlow dataset containing images and labels.
    batch_size (int): An integer representing the number of samples to include in each batch (default: BATCH_SIZE).
  Returns:
    tf.data.Dataset: A TensorFlow dataset configured for optimal performance.
  """
  ds = ds.cache()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  
  return ds

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

def calculate_mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, list[float]]:
    """
    Calculates the mean value of each metric across multiple folds of a K-fold cross validation for a machine learning model.

    Args:
        metrics_list (list[dict[str, float]]): A list of dictionaries containing the metrics for each fold of the
            K-fold cross validation. Each dictionary should have keys for each metric and values for each epoch.

    Returns:
        dict[str, list[float]]: A dictionary containing the mean value of each metric across all folds of the
            K-fold cross validation. The keys of the dictionary are the metric names, and the values are lists
            containing the mean value of the metric for each epoch.
    """

    # Initialise aggregate metrics
    aggregate_metrics: dict[str, list[float]] = {}
    for fold in metrics_list:
        for metric in fold.keys():
            if metric not in aggregate_metrics:
                aggregate_metrics[metric] = []

    # Calculate the average metric per epoch for every fold
    number_of_folds: int = len(metrics_list)
    for metric in aggregate_metrics.keys():
        number_of_epochs: int = len(metrics_list[0][metric])
        for epoch in range(number_of_epochs):
            # A list of every value for that given metric in this epoch across folds
            values_per_epoch: list[float] = [x[metric][epoch] for x in metrics_list]
            mean_per_epoch  : float = sum(values_per_epoch) / number_of_folds
            aggregate_metrics[metric].append(mean_per_epoch)

    return aggregate_metrics

def calculate_avg_max(k_fold_metrics: list[dict[str, float]], metric_name: str = "val_auc") -> float:
    """
    Calculates the maximum average metric value across all folds of a k-fold cross-validation experiment.

    Args:
        k_fold_metrics (list[dict[str, float]]): A list of dictionaries containing metric values for each fold.
            Each dictionary should contain the same set of keys representing the names of the metrics,
            and the corresponding values representing the metric values for the corresponding fold.
        metric_name (str, optional): The name of the metric for which the maximum average value is to be calculated.
            Defaults to "val_auc".
    Returns:
        float: The maximum average metric value across all folds.

    """
    avg_metrics: dict[str, list[float]] = calculate_mean_metrics(k_fold_metrics)
    max_avg_metric: float = max(avg_metrics[metric_name])

    return max_avg_metric

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

def k_fold_dataset(ds: tf.data.Dataset, k: int = 10) -> list[tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    Splits a TensorFlow dataset into k folds for k-fold cross-validation, and returns
    a list of tuples, where each tuple contains a training dataset and a validation dataset.

    Args:
        ds (tf.data.Dataset): A TensorFlow dataset containing the examples to be split into k folds.
        k (int, optional): The number of folds to create. Defaults to 10.

    Returns:
        A list of tuples, where each tuple contains a training dataset and a validation dataset.
    """
    
    # First shard the given dataset into k individual folds.
    list_of_folds   : list[tf.data.Dataset] = []
    for i in range(k):
        fold: tf.data.Dataset = ds.shard(num_shards=k, index=i)
        list_of_folds.append(fold)

    # Next, generate a list of train and validation dataset tuples
    list_of_ds_pairs: list[tuple[tf.data.Dataset, tf.data.Dataset]] = []
    for i, holdout_fold in enumerate(list_of_folds):
        ds_valid: tf.data.Dataset = holdout_fold

        # Use list slicing to select every fold except holdout_fold as the training folds
        training_folds: list[tf.data.Dataset] = list_of_folds[:i] + list_of_folds[i+1:]

        if len(training_folds) == 1:
            # If there is only one training fold (i.e. k=2), return that as ds_train
            ds_train: tf.data.Dataset = training_folds[0]

        else:
            # Else concatenate all the remaining folds to yield ds_train
            ds_train: tf.data.Dataset = training_folds[0]
            for fold in training_folds[1:]:
                ds_train = ds_train.concatenate(fold)

        # Batch, cache, and optimize for performance
        ds_train = configure_for_performance(ds_train)
        ds_valid = configure_for_performance(ds_valid)

        ds_pair: tuple[tf.data.Dataset, tf.data.Dataset] = (ds_train, ds_valid)
        list_of_ds_pairs.append(ds_pair)
    
    return list_of_ds_pairs

def cross_validate(
        ModelClass: tf.keras.Model,
        ds_train_and_valid: tf.data.Dataset,
        filename: str = "FINAL_k_fold_history.pickle",
        epochs: int = 50,
        batch_size: int = 128,
        dropout_rate: float = 0.0,
        k: int = 10,
        optimizer_kwargs: dict[str, any] = {"learning_rate": 0.001, "epsilon": 1e-7}
    ) -> list[tf.keras.callbacks.History]:
    """
    Performs K-fold cross-validation on a given dataset using a specified deep learning model.

    Args:
        ModelClass (tf.keras.Model): The class of the model to be trained and evaluated.
        ds_train_and_valid (tf.data.Dataset): The input dataset to be used for training and validation.
        metrics (list[tf.metrics.Metric]): A list of metrics to be evaluated during training.
        loss (tf.keras.losses.Loss, optional): The loss function to be used for training. Defaults to BinaryCrossentropy.
        optimizer (tf.keras.optimizers.Optimizer, optional): The optimizer to be used for training. Defaults to Adam().
        epochs (int, optional): The number of epochs to train for. Defaults to 50.
        batch_size (int, optional): The batch size to use during training. Defaults to 128.
        k (int, optional): The number of folds to use for K-fold cross-validation. Defaults to 10.

    Returns:
        list[tf.keras.callbacks.History]: A list of `tf.keras.callbacks.History` objects containing training metrics and loss values for each fold.
    """

    history_list: list[tf.keras.callbacks.History] = []
    train_valid_pairs  : list[tf.data.Dataset] = k_fold_dataset(ds_train_and_valid, k)

    for i, (ds_train, ds_valid) in enumerate(train_valid_pairs):

        ds_train_size: int = tf.data.experimental.cardinality(ds_train)
        ds_valid_size: int = tf.data.experimental.cardinality(ds_valid)
        print(f"Training fold {i + 1}/{k}: ds_train: {ds_train_size}, ds_valid: {ds_valid_size}")

        tf.keras.backend.clear_session()
        model = ModelClass(dropout_rate=dropout_rate)

        model_callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"FINAL_MODEL_CHECKPOINT_kfold_{i + 1}.model",
            monitor="val_auc",
            verbose=1,
            save_best_only=True,
            mode="max",
            save_freq="epoch"
        )

        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.AUC(multi_label=True, num_labels=18),
                tf.keras.metrics.Precision(thresholds=0.5),
                tf.keras.metrics.Recall(thresholds=0.5),
                tfa.metrics.F1Score(num_classes=18, average='macro', threshold=0.5),
            ]
        )
                
        history = model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[model_callback_checkpoint]
        )
        
        history_list.append(history.history)
        # Save results in case hyperparameter search gets interrupted
        with open(filename, 'wb') as file:
            pickle.dump(history_list, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"K-Fold Cross-Validation Completed on {i +1} Folds.")
    return history_list
