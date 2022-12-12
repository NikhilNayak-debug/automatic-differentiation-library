# test_AD.py
# Created at 2:15 PM 11/22/22 by Saket Joshi
# This test file contains all the test cases for fab_ad_tensor.py

import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/fab_ad')))
from fab_ad_tensor import FabTensor, AdMode
from fab_ad_session import fab_ad_session
from fab_ad_diff import auto_diff
from constants import *


def test_fabtensor_sanity():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    z = x ** 2 + y ** 2
    output = auto_diff(output=z, mode=AdMode.FORWARD)
    assert output.value == 25
    assert output.gradient[0] == 6
    assert output.gradient[1] == -8
    assert z.identifier == 'x^2 + y^2'
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    output = auto_diff(output=x, mode=AdMode.FORWARD)
    assert output.gradient == 1.0


def test_fabtensor_repr():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    assert repr(x) == "value: 3 derivative: [1. 0. 0.] name: x reverse mode gradient: 0"


def test_fabtensor_str():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, derivative=0, identifier='x')
    assert str(x) == "value: 3 derivative: [0] name: x reverse mode gradient: 0"


def test_fabtensor_equal():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=3, identifier='y')
    assert x == y
    assert x == 3


def test_fabtensor_not_equal():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=4, identifier='y')
    assert x != y
    assert x != 4


def test_fabtensor_inequalities():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=4, identifier='y')
    assert x < y
    assert x < 4
    assert x <= y
    assert x <= 4
    assert y > x
    assert y > 3
    assert y >= x
    assert y >= 3


def test_fabtensor_len():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, derivative=[1, 0], identifier='x')
    z = len(x)
    assert z == 2


def test_fabtensor_neg():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    z = -x
    assert z.value == -3
    output = auto_diff(output=z, mode=AdMode.FORWARD)
    assert output.gradient == -1

def test_fabtensor_add():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    z = x + y
    assert z.value == -1
    output = auto_diff(output=z, mode=AdMode.FORWARD)
    assert output.gradient[0] == 1
    assert output.gradient[1] == 1
    assert z.identifier == 'x + y'

    z = x + 5
    assert z.value == 8
    output1 = auto_diff(output=z, mode=AdMode.FORWARD)
    output2 = auto_diff(output=x, mode=AdMode.FORWARD)
    assert output1.gradient[0] == output2.gradient[0]
    assert z.identifier == 'x + 5'


def test_fabtensor_radd():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    z = 1 + x
    assert z.value == 4
    assert z.derivative[0] == 1.
    assert z.identifier == '1 + x'

    z = 5 + x
    assert z.value == 8
    assert z.derivative[0] == x.derivative[0]
    assert z.identifier == '5 + x'


def test_fabtensor_iadd():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    x += y
    assert x.value == -1
    assert x.derivative[0] == 1.
    assert x.identifier == 'x + y'

    with pytest.raises(TypeError):
        x += {1.0}


def test_fabtensor_sub():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    z = x - y
    assert z.value == 7
    assert z.derivative[1] == -1
    assert z.identifier == 'x - y'

    z = x - 5
    assert z.value == -2
    assert z.derivative[0] == x.derivative[0]
    assert z.identifier == 'x - 5'


def test_fabtensor_rsub():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    z = 1 - x
    assert z.value == -2
    assert z.derivative[0] == -1
    assert z.identifier == '1 - x'

    z = 5 - x
    assert z.value == 2
    assert z.derivative[0] == -1 * x.derivative[0]
    assert z.identifier == '5 - x'


def test_fabtensor_isub():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    x -= y
    assert x.value == 7
    assert x.derivative[1] == -1
    assert x.identifier == 'x - y'

    with pytest.raises(TypeError):
        x -= {1.0}


def test_fabtensor_mul():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    z = x * y
    assert z.value == -12
    assert z.derivative[0] == -4
    assert z.derivative[1] == 3
    assert z.identifier == 'x * y'

    z = x * 3
    assert z.value == 9
    assert z.derivative[0] == x.derivative[0] * 3
    assert z.identifier == 'x * 3'


def test_fabtensor_rmul():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    z = 1 * x
    assert z.value == 3
    assert z.derivative[0] == 1
    assert z.identifier == 'x'

def test_fabtensor_imul():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    x *= y
    assert x.value == -12
    assert x.derivative[0] == -4
    assert x.derivative[1] == 3
    assert x.identifier == 'x * y'

    with pytest.raises(TypeError):
        z = x * {2.0}

def test_fabtensor_truediv():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    z = x / y
    assert z.value == -0.75
    assert z.derivative[0] == -0.25
    assert z.derivative[1] == -0.1875
    assert z.identifier == 'x * 1 / y'

    z = x / 2
    assert z.value == 1.5
    assert z.derivative[0] == x.derivative[0] / 2


def test_fabtensor_rtruedeiv():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    z = 1 / x
    assert z.value == 0.3333333333333333
    assert z.derivative[0] == -0.1111111111111111
    assert z.identifier == '1 / x'
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=10.0, identifier='x')
    with pytest.raises(TypeError):
        z = x / {2.0}


def test_fabtensor_itruediv():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=-4, identifier='y')
    x /= y
    assert x.value == -0.75
    assert x.derivative[0] == -0.25
    assert x.derivative[1] == -0.1875
    assert x.identifier == 'x * 1 / y'
    with pytest.raises(TypeError):
        x /= {2.0}


def test_fabtensor_pow():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=4, identifier='y')
    z = x ** y
    assert z.value == 81
    z.derivative[0] == 108.0
    assert pytest.approx(z.derivative[1], 0.01) == 88.9875953821169
    assert z.identifier == 'x^y'
    with pytest.raises(TypeError):
        z = x ** {2.0}


def test_fabtensor_rpow():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier='x')
    z = 1 ** x
    assert z.value == 1
    assert z.derivative[0] == 0
    assert z.identifier == '1^x'


def test_fabtensor_directional_derivative():
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, derivative=[1, 0], identifier='x')
    y = FabTensor(value=-4, derivative=[0, 1], identifier='y')
    z = x * y
    assert z.directional_derivative(seed_vector=[1, 0]) == -4
    assert z.directional_derivative(seed_vector=[0, 1]) == 3


if __name__ == "__main__":
    pass
