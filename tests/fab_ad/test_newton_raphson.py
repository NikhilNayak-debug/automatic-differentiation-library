import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/fab_ad')))
from fab_ad_tensor import FabTensor, AdMode
from fab_ad_session import fab_ad_session
from fab_ad_diff import auto_diff
from constants import *


def func(x):
    z = x * x * x - x * x + 2
    return auto_diff(output=z, mode=AdMode.FORWARD).value


def derivFunc(x):
    z = x * x * x - x * x + 2
    return auto_diff(output=z, mode=AdMode.FORWARD).gradient


# Function to find the root
def newtonRaphson(x):
    fab_ad_session.initialize(num_inputs=3)
    tensor = FabTensor(value=x, identifier="x")
    h = func(tensor) / derivFunc(tensor)
    while True:
        if isinstance(h, float):
            if abs(h) < 0.0001:
                break
        else:
            if max(abs(h)) < 0.0001:
                break
        # x(i+1) = x(i) - f(x) / f'(x)
        x = x - h
        fab_ad_session.initialize(num_inputs=3)
        tensor = FabTensor(value=x, identifier="x")
        h = func(tensor) / derivFunc(tensor)

    print("The value of the root is : ", x)


# Driver program to test above
x0 = -20.00 # Initial values assumed
newtonRaphson(x0)

x0 = [-10.00, 10.00] # Initial values assumed
newtonRaphson(x0)