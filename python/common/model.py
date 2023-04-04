#!/usr/bin/env python3

import tensorflow as tf
from keras import layers
from typing import Union, Literal

RNG_SEED: int = 1337

class TransferLearningModel(tf.keras.Model):
    """
    Transfer Learning Model.
    """
    def __init__(
            self,
            name=None,
            weights: Literal['imagenet', 'radimagenet.h5'] = 'imagenet',
            dropout_rate: float = 0.0,
            **kwargs
        ):
        super().__init__(**kwargs)

        # First, we will define the different components of the model separately
        self.input_layer: tf.Tensor = layers.InputLayer(input_shape=(299, 299, 3), name="Input_Layer")
        self.data_augmentation: tf.keras.Sequential = tf.keras.Sequential(
            [
                layers.RandomFlip(seed=RNG_SEED),
            ],
            name="Data_Augmentation_Pipeline"
        )
        self.inceptionv3: tf.keras.Model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=weights,
            input_shape=(299, 299, 3)
        )

        # Freeze base model weights for transfer learning
        self.inceptionv3.trainable = False
        self.classifier: tf.keras.Sequential = tf.keras.Sequential(
            [
                layers.GlobalMaxPooling2D(),
                layers.Dense(1024, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense( 512, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense( 256, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(  18, activation='sigmoid')
            ],
            name="RUST_Score_Classifier"
        )

        # Finally, we define the model as the sum of it's components
        self.model: tf.keras.Sequential = tf.keras.Sequential(
            [
                self.input_layer,
                self.data_augmentation,
                self.inceptionv3,
                self.classifier
            ],
            name="InceptionV3_TransferLearning"
        )
    def call(self, inputs):
        return self.model(inputs)