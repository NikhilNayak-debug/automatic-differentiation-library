[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team32/actions/workflows/test.yml/badge.svg)](https://code.harvard.edu/CS107/team32/actions/workflows/test.yml)
[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team32/actions/workflows/coverage.yml/badge.svg)](https://code.harvard.edu/CS107/team32/actions/workflows/coverage.yml)

# team32
CS107/AC207 Project

# Introduction
Differentiation is defined as the process of finding the gradients/derivatives of a particular function in hand. It finds multiple applications in the areas of science and engineering. With the exponential growth in the size of the dataset and advancements in technologies - the complexity of computing derivatives has increased and we have become increasingly dependent on computers to compute derivatives.  

Currently, there are three ways to compute derivatives - finite, symbolic, automatic differentiation. The finite differentiation method although being quick and easy to implement - suffers from machine precision and rounding errors. We are able to alleviate these issues using symbolic differentiation, however, it becomes computationally very expensive as the function(s) starts to get complex. We are able to alleviate both the issues of computational complexity and machine precision using automatic differentiation.  

Automatic Differentiation leverages symbolic rules for evaluating gradients - which is more accurate than using finite difference approximations. But unlike a purely symbolic process, the evaluation of expressions takes place early in the computation - it evaluates derivatives at particular numeric values.  

The package `fab-ad` implements automatic differentiation for computational use. `fab-ad` can be used to automatically differentiate functions via forward mode. Automatic Differentiation finds applications in optimization, machine learning, and numerical methods.  

# Installation
Our package is for Python 3 only. You can access our package by cloning 
our repository. To clone run our repository, run `git clone 
git@code.harvard.edu:CS107/team32.git` from command line. Once you clone 
the repository you can use `cd team32` where you can find all the files. 
From there, you can use `cd src` to go where all the modules reside. Then 
use `python -m pip install toml` which will install all the requirements 
specified in toml.

## 3. How To Use `fab_AD`?

### 3.1 Package Installation, testing and import
#### 3.1.1 Installation from the source
Our package is for Python 3 only. You can access our package by cloning our repository. 

- To clone run our repository, run `git clone git@code.harvard.edu:CS107/team32.git` from command line. 
- Once you clone the repository you can use `cd team32` where you can find all the files. 
- From there, you can use `cd src` to go where all the modules reside. 
- Then use `python -m pip install poetry` and `poetry install` which will install all the requirements specified in toml.
- Run `poetry shell` to activate new virtual env.
```bash
git clone git@code.harvard.edu:CS107/team32.git
cd team32
python3 -m pip install poetry
poetry install
poetry shell
```

#### 3.1.2 Installation via PyPI

fab_AD is available at (https://test.pypi.org/simple/ fab-ad). You can download it by the command given below.
```bash
python3 -m venv test
source test/bin/activate
pip install numpy==1.23.5
pip install -i https://test.pypi.org/simple/ fab-ad
```

 # User Guide
Once you install the package, you can simple import it by `from fab_ad 
import FabTensor`.
Afterwards, you could initiate the FabTensor object by giving the point 
where you wish to differentiate. FabTensor can take in a vector input 
values, representing a point's coordinates in multi-dimensional space. 
Moreover, you could also add other supplementary features as in the code 
demo provided below. You can find this demo file by the name of usage.py 
under src in the github.

#### Usage: Forward Mode AD

```python 
from fab_ad.fab_ad_tensor import FabTensor, AdMode
from fab_ad.fab_ad_session import fab_ad_session
from fab_ad.fab_ad_diff import auto_diff
from fab_ad.constants import *

# multiple scalar input; single scalar output; forward ad

# initialize the fab_ad session with number of input variables. if unsure, set num_inputs to a high number (defaults to 10)
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
# Function 0: Value: 41
# Gradient w.r.t x = 6.0
# Gradient w.r.t y = -16.0
```

#### Usage: Reverse Mode AD

```python
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
# Function 0: Value: 41
# Gradient w.r.t x = 6.0
# Gradient w.r.t y = -16.0
```

#### Usage: Gradient Descent

```python
from fab_ad.fab_ad_tensor import FabTensor, AdMode
from fab_ad.fab_ad_session import fab_ad_session
from fab_ad.fab_ad_diff import auto_diff
from fab_ad.constants import *

def function_derivative(x: FabTensor, y: FabTensor):
    # compute output variable
    z = x**2 + y**4
    # compute gradient of output variable with respect to input variables
    return auto_diff(output=z, mode=AdMode.FORWARD).gradient

def gradient_descent(
    function_derivative, start, learn_rate, n_iter=10000, tolerance=1e-10
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
        
        if (i%1000) == 0:
            print(f"iteration {i}: {vector}")
    
    return vector


start = np.array([1.0, 1.0])
print(gradient_descent(function_derivative, start, 0.2, tolerance=1e-08).round(4))

# iteration 0: [0.6 0.2]
# iteration 1000: [8.49966157e-223 2.47685235e-002]
# iteration 2000: [4.9406565e-324 1.7593023e-002]
# iteration 3000: [4.94065646e-324 1.43868515e-002]
# iteration 4000: [4.94065646e-324 1.24691641e-002]
# iteration 5000: [4.9406565e-324 1.1158074e-002]
# iteration 6000: [4.94065646e-324 1.01891453e-002]
# iteration 7000: [4.94065646e-324 9.43548979e-003]
# iteration 8000: [4.94065646e-324 8.82762731e-003]
# iteration 9000: [4.94065646e-324 8.32389824e-003]
# [0.     0.0079]
```

#### Usage: Newton-Raphson

```python
import os
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
# The value of the root is :  -1.0000001181322415

x0 = [-10.00, 10.00] # Initial values assumed
newtonRaphson(x0)
# The value of the root is :  [-1.         -1.00000001]
```

# Documentation
You can find the documentation for this package [here](https://code.harvard.edu/pages/CS107/team32/).

We use `numpydoc==1.5.0`  and `sphinx==4.2` for documentation.

# Future Features
As part of the extension for this project we plan on implementing higher order and mixed derivatives of vector functions. Specifically we will implement the Hessian matrix which includes the derivative of a function twice w.r.t each independent variable (similar to the Laplacian operator) and mixed derivatives (i.e. derivative w.r.t one independent variable followed by another independent variable). Currently, we have implemented only the first derivative of a function w.r.t all the seed vectors/independent variables.

### Changes in software
Currently the derivative data member of FabTensor object is a 1D array. This would change to 2D array to accommodate the Hessian matrix. Also, we would have to update each of our elementary math functions to return the second derivatives by taking the derivative of the first derivative. The main challenge here will be to efficiently compute the second derivatives from the first derivatives by reusing most of the code already written and using a helper function to call the same derivative function twice.

### Changes in directory structure, new modules, classes, data structures, etc
Directory structure would remain the same. Modules and classes would also remain the same because we don't need a new class to implement the second derivative. We would need a new method in the fab_ad class to call the derivative for the second time (derivative of the first derivative). Also, we would have to update each of our elementary math functions to return the second derivatives. Data structure of the derivative member of the FabTensor class would change from being a 1D array to 2D array to store all the second derivatives i.e. the Hessian matrix.

Higher order and mixed derivatives, specifically second order derivatives can then be used to determine the concavity, convexity, the points of inflection, and local extrema of functions. As pointed out earlier it can also be used to compute the Laplacian operator which is useful in describing many different phenomena, from electric potentials, to the diffusion equation for heat and fluid flow, and quantum mechanics.

# Broader Impact, Inclusivity and Future directions
We hope that our package will be used in a variety of disciplines, 
including physics, engineering, applied mathematics, astronomy, and even 
domains that the package's creators could never have envisaged. We believe 
that this package can be used to perform automatic differentiations 
accurately and effectively as well as serve as a model for the future 
creation of improved automatic differentiation packages. We see several 
opportunities for this package to be improved and would be pleased to see 
them realized.

On the other hand, we don't want to see this program being used as a 
shortcut for differentiating work or for plagiarism or cheating. This 
package's open-source nature makes it available to everyone, but it also 
leaves it vulnerable to those who intend to use it for plagiarism. Users 
should be mindful of this nature and pick their method of using this 
package carefully. This software is not intended to be used as a shortcut 
for differentiation procedures. It could be used to check results for 
derivative calculations done manually or using other algorithms, although 
derivative calculation techniques should still be used instead. These 
exercises serve a purpose, and using this software to find the solutions 
does not advance learning.

Also evident is the connection between our automatic differentiation 
algorithms and mathematical concepts like the Leibniz Rule and the Faa di 
Bruno Formula that we made while working on this project. It shouldn't be 
the first time that these formulas have been employed to compute 
higher-order derivatives, but it provided motivation for us to carry out 
the implementation ourselves. In order to close the gap between theories 
and applications, we want for our project to serve as a case study. This 
experience demonstrates that now is the perfect time for all types of 
information to come together in order to support new discoveries, which 
will enable us and many students to continue working toward this aim.

# Software Inclusivity
Users and collaborators from all origins and identities are welcome to the 
fab_ad package and its developer. As was seen during the creation of this 
package, we think that trust, respect, and care are the foundations of 
greatness in a collaborative endeavor. In an effort to reach as many 
people as possible who are interested in this package, we made every 
effort to make it as inclusive and user-friendly as possible by including 
the necessary documentation and instructions. Although this package was 
created using Python and English, anyone who is proficient in another 
language or programming language is welcome to contribute. Pull requests 
are examined and approved by all developers while this package is being 
developed.

Every time one of us feels the need to start a pull request, that person 
would talk to the other members and come to a consensus. We would adore 
the opportunity to carry this constructive dialogue into other 
collaborations with this package.



<!-- ## Development environment setup

1. Create virtual environment

2. Use poetry to install Dependencies
```bash
python3 -m pip install poetry
poetry install
``` -->

### Running tests
<TO DO>
All tests can be run for the repository using below command:

```
python -m pytest tests/
```


### Making documentation
```
Populate your master file team32/index.rst and create other documentation source files. Use the Makefile to build the docs, like so:
   make builder
where "builder" is one of the supported builders, e.g. html, latex or linkcheck.

```


