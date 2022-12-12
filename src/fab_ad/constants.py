import numbers
import numpy as np
from enum import Enum


_ALLOWED_NUMERICS = (int, float, numbers.Integral, numbers.Number, np.ndarray)
_ALLOWED_ITERABLES = (list, np.ndarray)
_SPECIAL_FUNCTIONS = "sin, cos, tan, cosec, sec, cot, arcsin, arccos, arctan, arccosec, arcsec, arccot," \
                     " exp, sinh, cosh, tanh, cosech, sech, coth, logistic, log, sqrt"
_MAX_INDEPENDENT_VARS = 10
_GLOBAL_COUNTER = 0
