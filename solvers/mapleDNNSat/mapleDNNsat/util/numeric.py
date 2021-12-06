import numpy as np


def is_numeric_literal(obj):
    return isinstance(obj, float) or isinstance(obj, np.floating) or isinstance(obj, int)
