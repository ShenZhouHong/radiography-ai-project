#!/usr/bin/env python3

from .model import TransferLearningModel
from .utilities import *
from .datasetutils import *
from .plotting import *
from .kfold import k_fold_dataset
from .crossvalidate import cross_validate

def selfcheck() -> bool:
    """
    Test function to verify that Python imports were successful
    """
    print("Python imports were successful.")
    return True