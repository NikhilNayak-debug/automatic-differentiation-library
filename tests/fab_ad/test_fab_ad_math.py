# test_AD.py
# Created at 2:15 PM 11/22/22 by Saket Joshi
# This test file contains all the test cases for fab_ad_tensor.py

import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/fab_ad')))
from fab_ad_tensor import FabTensor, AdMode
from fab_ad_diff import auto_diff
from fab_ad_session import fab_ad_session
from fab_ad_math import *
from constants import *


def test_fabtensor_sqrt():
    fab_ad_session.clear()
    x = FabTensor(value=4, identifier='x')
    y = FabTensor(value=9, identifier='y')
    z = sqrt(x) + sqrt(y)
    assert pytest.approx(z.value, 0.01) == 5.0
    assert z.derivative[0] == 0.25
    assert z.derivative[1] == 1 / 6
    assert z.identifier == 'x^0.5 + y^0.5'
    z = sqrt(4)
    assert z == 2


def test_fabtensor_exp():
    fab_ad_session.clear()
    x = FabTensor(value=4, identifier='x')
    y = FabTensor(value=9, identifier='y')
    z = exp(x) + exp(y)
    assert pytest.approx(z.value, 0.01) == 8157.682077608529
    assert pytest.approx(z.derivative[0], 0.01) == 54.598150033144236
    assert pytest.approx(z.derivative[1], 0.01) == 8103.083927575384
    assert z.identifier == '2.718281828459045^x + 2.718281828459045^y'
    z = exp(1)
    assert z == np.exp(1)


def test_fabtensor_log():
    fab_ad_session.clear()
    tensor = FabTensor(value=4, identifier='x')
    z = log(tensor) + log(tensor)
    assert pytest.approx(z.value, 0.01) == 2.772588722239781
    assert z.derivative[0] == 0.5
    assert z.identifier == 'log(x) + log(x)'
    z = log(np.exp(1))
    assert pytest.approx(z.value, 0.01) == 1
    with pytest.raises(TypeError):
        log({1.0})
    with pytest.raises(ValueError):
        log(-1)


def test_fabtensor_sin():
    fab_ad_session.clear()
    x = FabTensor(value=4, identifier='x')
    z = sin(x) + sin(x)
    assert pytest.approx(z.value, 0.01) == -1.5136049906158566
    assert pytest.approx(z.derivative[0], 0.01) == -1.3072872417272239
    assert z.identifier == 'sin(x) + sin(x)'
    z = sin(np.pi / 2)
    assert z.value == 1
    with pytest.raises(TypeError):
        sin({0.0})


def test_fabtensor_cos():
    fab_ad_session.clear()
    x = FabTensor(value=4, identifier='x')
    y = FabTensor(value=9, identifier='y')
    z = cos(x) + cos(y)
    assert pytest.approx(z.value, 0.01) == -1.5647738827482889
    assert pytest.approx(z.derivative[0]) == 0.7568025
    assert pytest.approx(z.derivative[1]) == -0.41211849
    assert z.identifier == 'cos(x) + cos(y)'
    z = cos(np.pi / 2)
    assert pytest.approx(z.value, 0.00001) == 0
    with pytest.raises(TypeError):
        cos({0.0})


def test_fabtensor_tan():
    fab_ad_session.clear()
    x = FabTensor(value=4, identifier='x')
    y = FabTensor(value=9, identifier='y')
    z = tan(x) + tan(y)
    assert pytest.approx(z.value, 0.0001) == 0.7055056229077676
    assert pytest.approx(z.derivative[0], 0.0001) == 2.34055012
    assert pytest.approx(z.derivative[1], 0.0001) == 1.20458946
    assert z.identifier == 'tan(x) + tan(y)'
    z = tan(np.pi / 4)
    assert pytest.approx(z.value, 0.00001) == 1
    with pytest.raises(TypeError):
        tan({0.0})


def test_fabtensor_asin():
    fab_ad_session.clear()
    x = FabTensor(value=0.5, identifier='x')
    y = FabTensor(value=0.5, identifier='y')
    z = arcsin(x) + arcsin(y)
    assert pytest.approx(z.value, 0.01) == 1.0471975511965979
    assert pytest.approx(z.derivative[0], 0.0001) == 1.15470054
    assert pytest.approx(z.derivative[1], 0.0001) == 1.15470054
    assert z.identifier == 'sin^-1(x) + sin^-1(y)'
    z = arcsin(1 / 2)
    assert pytest.approx(z.value, 0.00001) == np.pi / 6
    with pytest.raises(ValueError):
        arcsin(2)
    with pytest.raises(TypeError):
        arcsin({0.0})


def test_fabtensor_arccos():
    fab_ad_session.clear()
    x = FabTensor(value=0.5 * (3 ** 0.5), identifier='x')
    y = FabTensor(value=0.5 * (3 ** 0.5), identifier='y')
    z = arccos(x) + arccos(y)
    assert pytest.approx(z.value, 0.01) == 1.0471975511965979 * 2
    assert pytest.approx(z.derivative[0], 0.0001) == -2
    assert pytest.approx(z.derivative[1], 0.0001) == -2
    assert z.identifier == 'cos^-1(x) + cos^-1(y)'
    z = arccos(1 / 2)
    assert pytest.approx(z.value, 0.00001) == np.pi / 3
    with pytest.raises(ValueError):
        arccos(2)
    with pytest.raises(TypeError):
        arccos({0.0})


def test_fabtensor_arctan():
    fab_ad_session.clear()
    x = FabTensor(value=3 ** -0.5, identifier='x')
    y = FabTensor(value=3 ** -0.5, identifier='y')
    z = arctan(x) + arctan(y)
    assert pytest.approx(z.value, 0.01) == 1.0471975511965976
    assert pytest.approx(z.derivative[0], 0.0001) == 0.75
    assert pytest.approx(z.derivative[1], 0.0001) == 0.75
    assert z.identifier == 'tan^-1(x) + tan^-1(y)'
    z = arctan(1)
    assert pytest.approx(z.value, 0.00001) == np.pi / 4
    with pytest.raises(TypeError):
        arctan({0.0})


def test_fabtensor_logistic():
    fab_ad_session.clear()
    x = FabTensor(value=3, identifier='x')
    y = FabTensor(value=4, identifier='y')
    z = logistic(x) + logistic(y)
    assert pytest.approx(z.value, 0.01) == 1.934587916860342
    assert pytest.approx(z.derivative[0], 0.0001) == 0.04517666
    assert pytest.approx(z.derivative[1], 0.0001) == 0.01766271
    z = logistic(1)
    assert pytest.approx(z.value, 0.00001) == np.exp(1) / (1 + np.exp(1))
    with pytest.raises(TypeError):
        logistic({0.0})


if __name__ == "__main__":

    test_fabtensor_sqrt()
    test_fabtensor_exp()
    test_fabtensor_log()
    test_fabtensor_sin()
    test_fabtensor_cos()
    test_fabtensor_tan()
    test_fabtensor_asin()
    test_fabtensor_arccos()
    test_fabtensor_arctan()
    test_fabtensor_logistic()