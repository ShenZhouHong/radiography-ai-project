#!/usr/bin/env python3
import tensorflow as tf

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
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  
  return ds
