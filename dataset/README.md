# RUST-AI Radiography Project Dataset

This directory will contain the raw dataset of radiographs with their attached RUST labels. At this moment, the radiograph images are not yet cleared for release.

## Dataset

The dataset is made available as three `tf.data.Dataset` objects which can be imported using tensorflow's `tf.data.Dataset.load()` method.

* `ds_train/`
* `ds_valid/`
* `ds_test/`

### Features

Within the `tf.data.Dataset` objects (i.e. the `ds_*` directories), features are image tensors of shape `(299, 299, 3)`. Further information to be added.

### Labels

Within the `tf.data.Dataset` objects (i.e. the `ds_*` directories), labels are available as one-hot encoded `tf.Tensor` objects of shape `(18,)`. 

Additionally, the labels are also available as separate one-hot and categorical encoded Pandas dataframes, exported as `.csv` files.

* `df_labels-onehot.csv`
* `df_labels-categorical.csv`

Further information to be added.

## Dataset Manifest

The three dataset objects are made with a train-test-validation split of 70%, 15%, 15% by default. This means that the data is already suitable for a classical holdout testing and validation training process.

* Total (`ds_train + ds_valid + ds_test`): 2936 (100%)
    * Training set (`ds_train`): 2054 (70%)
    * Hold-out Validation Set (`ds_valid`): 441 (15%)
    * Hold-out Test Set (`ds_test`):  441 (15%)

However, this project uses k-Fold validation. Hence, in practice the `ds_train` and `ds_valid` sets are concatenated together before use, and then the resulting combined `ds_train_and_valid` dataset is split into `k` folds for cross-validation. There are two different k-values used in this project: `k = 10` for fine-grained evaluation, and `k = 6` for hyperparameter searching.

* Total (`ds_train + ds_valid + ds_test`): 2936 (100%)
    * Training and Validation Set (`ds_train + ds_valid`): 2490 (85%):
        * K-Fold Cross-Validation, K = 10:
            * Validation Set: 249  (~8.5% per fold)
            * Training Set  : 2241 (~76%)
        * K-Fold Cross-Validation, K = 6:
            * Validation Set: 415  (~15% per fold)
            * Training Set  : 2075 (~70%)
    * Hold-out Validation Set (`ds_valid`): 441 (15%)
    * Hold-out Test Set (`ds_test`):  441 (15%)