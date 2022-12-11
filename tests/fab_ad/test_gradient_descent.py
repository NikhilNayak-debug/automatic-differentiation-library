import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/fab_ad')))
from fab_ad_tensor import FabTensor, AdMode
from fab_ad_session import fab_ad_session
from fab_ad_diff import auto_diff
from constants import *

def function_derivative(x: FabTensor, y: FabTensor):
    z = x**2 + y**4
    return auto_diff(output=z, mode=AdMode.FORWARD).gradient

def gradient_descent(
    function_derivative, start, learn_rate, n_iter=50, tolerance=1e-06
):
    vector = start
    for _ in range(n_iter):
        fab_session.clear()
        x = FabTensor(value=vector[0], identifier="x")
        y = FabTensor(value=vector[1], identifier="y")
        diff = -learn_rate * function_derivative(x, y)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


start = np.array([1.0, 1.0])
print(gradient_descent(function_derivative, start, 0.2, tolerance=1e-08))
