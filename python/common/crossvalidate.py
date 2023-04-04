#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
import pickle

def cross_validate(
        ModelClass: tf.keras.Model,
        ds_train_and_valid: tf.data.Dataset,
        epochs: int = 50,
        batch_size: int = 128,
        save_history: bool = True,
        history_filename: str = "model_kfold_history.pickle",
        checkpoint: bool = False,
        checkpoint_name: str = "model",
        k: int = 10,
        optimizer_kwargs: dict[str, any] = {"learning_rate": 0.001, "epsilon": 1e-7},
        model_kwargs: dict[str, any] = {"dropout_rate": 0.0, "weights": "imagenet"}
    ) -> list[tf.keras.callbacks.History]:
    """
    Performs k-fold cross-validation on a given dataset using a specified deep learning model.

    Args:
        ModelClass (tf.keras.Model): The model class to be used for cross-validation.
        ds_train_and_valid (tf.data.Dataset): The training and validation dataset for cross-validation.
        epochs (int, optional): The number of epochs for training the model. Defaults to 50.
        batch_size (int, optional): The batch size for training the model. Defaults to 128.
        dropout_rate (float, optional): The dropout rate for the model. Defaults to 0.0.
        save_history (bool, optional): Whether or not to save the training history. Defaults to True.
        history_filename (str, optional): The filename to use for saving the training history. Defaults to "model_kfold_history.pickle".
        checkpoint (bool, optional): Whether or not to use a checkpoint callback. Defaults to False.
        checkpoint_name (str, optional): The name to use for the checkpoint. Defaults to "model".
        k (int, optional): The number of folds for cross-validation. Defaults to 10.
        optimizer_kwargs (dict[str, any], optional): The optimizer keyword arguments to use. Defaults to {"learning_rate": 0.001, "epsilon": 1e-7}.
        model_kwargs (dict[str, any], optional): The model keyword arguments to use. Defaults to {"dropout_rate": 0.0, "weights": "imagenet"}.

    Returns:
        list[tf.keras.callbacks.History]: The training history for each fold of the cross-validation.
    """

    history_list: list[tf.keras.callbacks.History] = []
    train_valid_pairs  : list[tf.data.Dataset] = k_fold_dataset(ds_train_and_valid, k)

    for i, (ds_train, ds_valid) in enumerate(train_valid_pairs):

        ds_train_size: int = tf.data.experimental.cardinality(ds_train)
        ds_valid_size: int = tf.data.experimental.cardinality(ds_valid)
        print(f"Training fold {i + 1}/{k}: ds_train: {ds_train_size}, ds_valid: {ds_valid_size}")

        tf.keras.backend.clear_session()
        model = ModelClass(**model_kwargs)

        if checkpoint:
            callbacks: list[tf.keras.callbacks.Callback] = [tf.keras.callbacks.ModelCheckpoint(
                f"{checkpoint_name}_kfold_{i + 1}",
                monitor="val_auc",
                save_best_only=True,
                mode="max",
                save_freq="epoch",
                verbose=0
            )]
        else:
            callbacks: list[tf.keras.callbacks.Callback] = []
        
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
            callbacks=callbacks
        )
        
        history_list.append(history.history)
        if save_history:
            with open(history_filename, 'wb') as file:
                pickle.dump(history_list, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"K-Fold Cross-Validation Completed on {i +1} Folds.")
    return history_list
