import numpy as np
from fab_ad_tensor import FabTensor, AdMode
from fab_ad_session import fab_ad_session
from fab_ad_diff import auto_diff
from constants import *


# scalar input; scalar output; forward ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
z = x ** 2
print(auto_diff(z, mode=AdMode.FORWARD))

# multiple scalar input; scalar output; forward ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
y = FabTensor(value=-4, identifier="y")
z = x ** 2 + 2 * y ** 2
print(auto_diff(z, mode=AdMode.FORWARD))

# scalar input; multiple scalar output; forward ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
functions = [
    x ** 2 + 2 * x + 1,
    x ** 3 - 8 * x
]
print(auto_diff(functions, mode=AdMode.FORWARD))

# multiple scalar input; multiple scalar output; forward ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
y = FabTensor(value=-4, identifier="y")
functions = [
    x ** 2 + 2 * x + 1,
    x ** 2 + 2 * y ** 2
]
print(auto_diff(functions, mode=AdMode.FORWARD))

# vector input; vector output; forward ad
fab_ad_session.clear()
x = FabTensor(value=[3, 4, -6], identifier="x")

z = x ** 2 + 2 * x + 1
print(auto_diff(z, mode=AdMode.FORWARD))

# multiple vector input; vector output; forward ad
fab_ad_session.clear()
x = FabTensor(value=[3, 4, -6], identifier="x")
y = FabTensor(value=[1, 2, 4], identifier="y")
z = x ** 2 + y ** 2
print(auto_diff(z, mode=AdMode.FORWARD))

# multiple vector input; multiple vector output; forward ad
fab_ad_session.clear()
x = FabTensor(value=[3, 4, -6], identifier="x")
y = FabTensor(value=[1, 2, 4], identifier="y")
functions = [
    x ** 2 + 2 * x + 1,
    x ** 2 + 2 * y ** 2
]
print(auto_diff(functions, mode=AdMode.FORWARD))

# scalar input; scalar output; reverse mode ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
z = x ** 2 + 2 * x + 1
print(auto_diff(z, mode=AdMode.REVERSE))

# multiple scalar input; scalar output; reverse ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
y = FabTensor(value=-4, identifier="y")
z = x ** 2 + 2 * y ** 2
print(auto_diff(z, mode=AdMode.REVERSE))

# scalar input; multiple scalar output; forward ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
functions = [
    x ** 2 + 2 * x + 1,
    x ** 3 - 8 * x
]
print(auto_diff(functions, mode=AdMode.FORWARD))

# multiple scalar input; multiple scalar output; forward ad
fab_ad_session.clear()
x = FabTensor(value=3, identifier="x")
y = FabTensor(value=-4, identifier="y")
functions = [
    x ** 2 + 2 * x + 1,
    x ** 2 + 2 * y ** 2
]
print(auto_diff(functions, mode=AdMode.FORWARD))
