# Analysis

The `analysis/` directory contains both the raw data collected from model training and hyperparameter search, as well as a series of 'lab' Jupyter notebooks that are used to explore and understand the data. These lab notebooks are self-contained and complete, however they are not the final product of the project.

## Raw Data

The raw data collected from the models are available as Python `pickle` files. These can be read by the `pickle` library directly as Python objects.

## Processed Data

The analysis notebooks output processed data from the raw pickle files. The processed data is made available as CSV files. These are used in the `latex/` directory to build graphs.