#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/12/2020, 13:21. Copyright (c) David J Turner

import numpy as np


def straight_line(x: np.ndarray, m: float, c: float):
    return m * x + c


def power_law(x: np.ndarray, k: float, a: float):
    return np.power(x, k) * a


# So that things like fitting functions can be written generally to support different models
MISC_MODELS = {}
MISC_MODELS_STARTS = {}






