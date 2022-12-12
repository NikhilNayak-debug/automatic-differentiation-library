import numpy as np
import pytest

from fab_ad_tensor import FabTensor, AdMode
from fab_ad_session import fab_ad_session
from fab_ad_diff import auto_diff
from constants import *


def test_ad():
    # scalar input; scalar output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    z = x ** 2
    result = auto_diff(z, mode=AdMode.FORWARD)
    assert result.value == 9
    assert result.gradient == 6
    print(result)

    # multiple scalar input; scalar output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    y = FabTensor(value=-4, identifier="y")
    z = x ** 2 + 2 * y ** 2
    result = auto_diff(z, mode=AdMode.FORWARD)
    assert result.value == 41
    assert all(result.gradient == np.array([6, -16]))
    print(result)

    # scalar input; multiple scalar output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    functions = [
        x ** 2 + 2 * x + 1,
        x ** 3 - 8 * x
    ]
    result = auto_diff(functions, mode=AdMode.FORWARD)
    assert all(result.value == [16, 3])
    assert all(result.gradient == np.array([8, 19]))
    print(result)

    # multiple scalar input; multiple scalar output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    y = FabTensor(value=-4, identifier="y")
    functions = [
        x ** 2 + 2 * x + 1,
        x ** 2 + 2 * y ** 2
    ]
    result = auto_diff(functions, mode=AdMode.FORWARD)
    assert all(result.value == [16, 41])
    assert all(result.gradient[0] == np.array([8, 0]))
    assert all(result.gradient[1] == np.array([6, -16]))
    print(result)

    # vector input; vector output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=[3, 4, -6], identifier="x")
    z = x ** 2 + 2 * x + 1
    result = auto_diff(z, mode=AdMode.FORWARD)
    assert all(result.value == [16, 25, 25])
    assert all(result.gradient == np.array([8, 10, -10]))
    print(result)

    # multiple vector input; vector output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=[3, 4, -6], identifier="x")
    y = FabTensor(value=[1, 2, 4], identifier="y")
    z = x ** 2 + y ** 2
    result = auto_diff(z, mode=AdMode.FORWARD)
    assert all(result.value == [10, 20, 52])
    assert all(result.gradient[0] == np.array([6, 8, -12]))
    assert all(result.gradient[1] == np.array([2, 4, 8]))
    print(result)

    # multiple vector input; multiple vector output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=[3, 4, -6], identifier="x")
    y = FabTensor(value=[1, 2, 4], identifier="y")
    functions = [
        x ** 2 + 2 * x + 1,
        x ** 2 + 2 * y ** 2
    ]
    result = auto_diff(functions, mode=AdMode.FORWARD)
    assert all(result.value[0] == [16, 25, 25])
    assert all(result.value[1] == [11, 24, 68])
    assert all(result.gradient[0][0] == np.array([8, 10, -10]))
    assert all(result.gradient[0][1] == np.array([0, 0, 0]))
    assert all(result.gradient[1][0] == np.array([6, 8, -12]))
    assert all(result.gradient[1][1] == np.array([4, 8, 16]))
    print(result)

    # scalar input; scalar output; reverse mode ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    z = x ** 2 + 2 * x + 1
    result = auto_diff(z, mode=AdMode.REVERSE)
    assert result.value == 16
    assert result.gradient == 8
    print(result)

    # multiple scalar input; scalar output; reverse ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    y = FabTensor(value=-4, identifier="y")
    z = x ** 2 + 2 * y ** 2
    result = auto_diff(z, mode=AdMode.REVERSE)
    assert result.value == 41
    assert all(result.gradient == np.array([6, -16]))
    print(result)

    # scalar input; multiple scalar output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    functions = [
        x ** 2 + 2 * x + 1,
        x ** 3 - 8 * x
    ]
    result = auto_diff(functions, mode=AdMode.FORWARD)
    assert all(result.value == np.array([16, 3]))
    assert all(result.gradient == np.array([8, 19]))
    print(result)

    # multiple scalar input; multiple scalar output; forward ad
    fab_ad_session.initialize(num_inputs=3)
    x = FabTensor(value=3, identifier="x")
    y = FabTensor(value=-4, identifier="y")
    functions = [
        x ** 2 + 2 * x + 1,
        x ** 2 + 2 * y ** 2
    ]
    result = auto_diff(functions, mode=AdMode.FORWARD)
    assert all(result.value == [16, 41])
    assert all(result.gradient[0] == np.array([8, 0]))
    assert all(result.gradient[1] == np.array([6, -16]))

    # test TypeError exception
    fab_ad_session.initialize(num_inputs=3)
    with pytest.raises(TypeError):
        fab_ad_session.initialize_derivative(value=None)

    # test TypeError exception
    fab_ad_session.initialize(num_inputs=3)
    with pytest.raises(TypeError):
        auto_diff(None, mode=AdMode.FORWARD)

    # test TypeError exception
    fab_ad_session.initialize(num_inputs=3)
    with pytest.raises(TypeError):
        auto_diff(None, mode=AdMode.REVERSE)


if __name__ == "__main__":
    test_ad()
