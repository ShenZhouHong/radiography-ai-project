#!/usr/bin/env python3

from .model import TransferLearningModel
from .utilities import *

def selfcheck() -> bool:
    """
    Test function to verify that Python imports were successful
    """
    print("Python imports were successful.")
    return True