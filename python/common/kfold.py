#!/usr/bin/env python3

import tensorflow as tf
from .datasetutils import configure_for_performance

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
