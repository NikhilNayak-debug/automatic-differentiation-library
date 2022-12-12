examples = {
    "Forward mode AD Example":
"""from fab_ad.fab_ad_tensor import FabTensor, AdMode
from fab_ad.fab_ad_session import fab_ad_session
from fab_ad.fab_ad_diff import auto_diff
from fab_ad.constants import *

# multiple scalar input; single scalar output; forward ad
# initialize the fab_ad session with number of input variables. if unsure, set num_inputs to a high number
fab_ad_session.initialize()
# define the input variables
x = FabTensor(value=3, identifier="x")
y = FabTensor(value=-4, identifier="y")
# compute the output variable
z = x ** 2 + 2 * y ** 2
# compute the gradient of the output variable with respect to the input variables
result = auto_diff(z, mode=AdMode.FORWARD)
assert result.value == 41
assert all(result.gradient == np.array([6, -16]))
print(result)

""",

"Reverse Mode AD Example":
"""
from fab_ad.fab_ad_tensor import FabTensor, AdMode
from fab_ad.fab_ad_session import fab_ad_session
from fab_ad.fab_ad_diff import auto_diff
from fab_ad.constants import *

# Multiple scalar input; scalar output; reverse ad
# initialize fab_ad session with number of input variables. if unsure, set num_inputs to a high number
fab_ad_session.initialize()
# initialize input variables
x = FabTensor(value=3, identifier="x")
y = FabTensor(value=-4, identifier="y")
# compute output variable
z = x ** 2 + 2 * y ** 2
# compute gradient of output variable with respect to input variables via reverse mode AD
result = auto_diff(z, mode=AdMode.REVERSE)

assert result.value == 41
assert all(result.gradient == np.array([6, -16]))
print(result)
""",
    
    "Gradient Descent":
"""from fab_ad.fab_ad_tensor import FabTensor, AdMode
from fab_ad.fab_ad_session import fab_ad_session
from fab_ad.fab_ad_diff import auto_diff
from fab_ad.constants import *

def function_derivative(x: FabTensor, y: FabTensor):
    # compute output variable
    z = x**2 + y**4
    # compute gradient of output variable with respect to input variables
    return auto_diff(output=z, mode=AdMode.FORWARD).gradient

def gradient_descent(
    function_derivative, start, learn_rate, n_iter=2000, tolerance=1e-10
):
    # initialize the vector
    vector = start
    # initialize the fab_ad session with number of input variables. if unsure, set num_inputs to a high number
    fab_ad_session.initialize()
    # initialize the input variables
    x = FabTensor(value=vector[0], identifier="x")
    y = FabTensor(value=vector[1], identifier="y")
    for i in range(n_iter):
        # compute the gradient descent step
        diff = -learn_rate * function_derivative(x, y)
        if np.all(np.abs(diff) <= tolerance):
            break
        # update the vector
        vector += diff
        # update the input variables
        x += diff[0]
        y += diff[1]
        
        if (i%200) == 0:
            print(f"iteration {i}: {vector}")
    
    return vector


start = np.array([1.0, 1.0])
print(gradient_descent(function_derivative, start, 0.2, tolerance=1e-08).round(4))

""",

"Newton-Raphson":
"""import os
import sys
import numpy as np

from fab_ad.constants import *
from fab_ad.fab_ad_tensor import FabTensor, AdMode
from fab_ad.fab_ad_session import fab_ad_session
from fab_ad.fab_ad_diff import auto_diff

# Function to find the root
def newtonRaphson(x):
    
    def func(x):
        # compute output variable and return value
        z = x * x * x - x * x + 2
        return auto_diff(output=z, mode=AdMode.FORWARD).value
    
    def derivFunc(x):
        # compute output variable and return gradient
        z = x * x * x - x * x + 2
        return auto_diff(output=z, mode=AdMode.FORWARD).gradient
    
    # initialize the fab_ad session with number of input variables. if unsure, set num_inputs to a high number
    fab_ad_session.initialize()
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
        tensor = tensor - h
        h = func(tensor) / derivFunc(tensor)

    print("The value of the root is : ", x)


# Driver program to test above
x0 = -20.00 # Initial values assumed
newtonRaphson(x0)

x0 = [-10.00, 10.00] # Initial values assumed
newtonRaphson(x0)

""",

}