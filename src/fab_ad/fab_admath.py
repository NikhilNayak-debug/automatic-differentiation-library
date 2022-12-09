import math
import numbers
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from fab_ad import FabTensor
from constants import _ALLOWED_TYPES, _SPECIAL_FUNCTIONS


def compute_gradient_reverse_mode(tensor: FabTensor, current_gradient=1):
    for child_tensor, local_gradient in tensor.children:
        child_tensor.gradient += current_gradient * local_gradient
        compute_gradient_reverse_mode(child_tensor, )


def sin(tensor):
    """sin of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        sin of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        return FabTensor(
            value=np.sin(tensor.value),
            derivative=np.cos(tensor.value) * tensor.derivative,
            identifier=f"sin({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, np.cos(tensor.value))
            ])
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=np.sin(tensor), derivative = 0, identifier="sin(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def cos(tensor):
    """cos of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cos of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        return FabTensor(
            value=np.cos(tensor.value),
            derivative=-1 * np.sin(tensor.value) * tensor.derivative,
            identifier=f"cos({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, -np.sin(tensor.value))
            ])
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=np.cos(tensor), derivative=0, identifier="cos(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def tan(tensor):
    """tan of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        tan of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        return FabTensor(
            value=np.tan(tensor.value),
            derivative=(1 / (np.cos(tensor.value) ** 2)) * tensor.derivative,
            identifier=f"tan({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, (1 / (np.cos(tensor.value) ** 2)))
            ])
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=np.tan(tensor), derivative=0, identifier="tan(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def cosec(tensor):
    return 1 / sin(tensor)


def sec(tensor):
    return 1 / cos(tensor)


def cot(tensor):
    return 1 / tan(tensor)


def arcsin(tensor):
    """sin inverse of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        sin inverse of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        if not (-1 <= tensor.value <= 1):
            raise ValueError("Value of tensor out of range for function arcsin!")
        return FabTensor(
            value=np.arcsin(tensor.value),
            derivative=(1 / ((1 - tensor.value ** 2) ** 0.5)) * tensor.derivative,
            identifier=f"sin^{-1}({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, 1 / ((1 - tensor.value ** 2) ** 0.5)),
            ]
        )
    elif isinstance(tensor, _ALLOWED_TYPES):
        if not (-1 <= tensor <= 1):
            raise ValueError("Value of tensor out of range for function arcsin!")
        return FabTensor(value=np.arcsin(tensor), derivative = 0, identifier="sin^{-1}(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def arccos(tensor):
    """cos inverse of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cos inverse of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        if not (-1 <= tensor.value <= 1):
            raise ValueError("Value of tensor out of range for function arccos!")
        return FabTensor(
            value=np.arcsin(tensor.value),
            derivative=(-1 / ((1 - tensor.value ** 2) ** 0.5)) * tensor.derivative,
            identifier=f"cos^{-1}({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, -1 / ((1 - tensor.value ** 2) ** 0.5))
            ]
        )
    elif isinstance(tensor, _ALLOWED_TYPES):
        if not (-1 <= tensor <= 1):
            raise ValueError("Value of tensor out of range for function arccos!")
        return FabTensor(value=np.arccos(tensor), derivative = 0, identifier="cos^{-1}(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def arctan(tensor):
    """tan inverse of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        tan inverse of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        return FabTensor(
            value=np.arctan(tensor.value),
            derivative=(1 / (1 + tensor.value ** 2)) * tensor.derivative,
            identifier=f"tan^{-1}({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, 1 / (1 + tensor.value ** 2)),
            ]
        )
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=np.arctan(tensor), derivative = 0, identifier="tan^{-1}(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def arccosec(tensor):
    """cosec inverse of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cosec inverse of tensor with updated value and derivative
    """
    return arcsin(1 / tensor)


def arcsec(tensor):
    """sec inverse of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        sec inverse of tensor with updated value and derivative
    """
    return arccos(1 / tensor)


def arccot(tensor):
    """cot inverse of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cot inverse of tensor with updated value and derivative
    """
    return arctan(1 / tensor)


def exp(tensor, base=np.e):
    """exponential of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        exponential of tensor with updated value and derivative

    """
    return base ** tensor


def sinh(tensor):
    """sinh of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        sinh of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        return FabTensor(
            value=np.sinh(tensor.value),
            derivative=np.cosh(tensor.value) * tensor.derivative,
            identifier=f"sinh({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, np.cosh(tensor.value)),
            ]
        )
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=np.sinh(tensor), derivative=0, identifier="sinh(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def cosh(tensor):
    """cosh of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cosh of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        return FabTensor(
            value=np.cosh(tensor.value),
            derivative=np.sinh(tensor.value) * tensor.derivative,
            identifier=f"cosh({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, np.sinh(tensor.value)),
            ]
        )
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=np.cosh(tensor), derivative=0, identifier="cosh(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def tanh(tensor):
    """cosh of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cosh of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        return FabTensor(
            value=np.tanh(tensor.value),
            derivative=(1 / np.cosh(tensor.value) ** 2) * tensor.derivative,
            identifier=f"tanh({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, 1 / np.cosh(tensor.value) ** 2),
            ]
        )
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=np.cosh(tensor), derivative=0, identifier="tanh(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def cosech(tensor):
    return 1 / sinh(tensor)


def sech(tensor):
    return 1 / cosh(tensor)


def coth(tensor):
    return 1 / tanh(tensor)


def logistic(tensor):
    """logistic of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        logistic of tensor with updated value and derivative

    """
    if isinstance(tensor, FabTensor):
        return 1 / (1 + exp(-tensor))
    elif isinstance(tensor, _ALLOWED_TYPES):
        return FabTensor(value=1 / (1 + np.exp(-tensor)), derivative=0, identifier=f"logistic({tensor})")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def log(tensor, base=np.e):
    """natural logarithm of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor
    base : float

    Returns
    -------
    FabTensor
        natural log of tensor with updated value and derivative
    """
    if isinstance(tensor, FabTensor):
        if tensor.value < 0:
            raise ValueError("Cannot compute logarithm for FabTensor with negative value!")
        return FabTensor(
            value=np.log(tensor.value),
            derivative=(1.0 / tensor.value) * tensor.derivative * (1 / np.log(base)),
            identifier=f"log({tensor.identifier})",
            mode=tensor.mode,
            source=[
                (tensor, 1.0 / (tensor.value * np.log(base))),
            ]
        )
    elif isinstance(tensor, _ALLOWED_TYPES):
        if tensor < 0.0:
            raise ValueError("Value of tensor out of range for function log!")
        return FabTensor(value=np.log(tensor), derivative=0, identifier="log(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_TYPES} only!")


def sqrt(tensor):
    """square root of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        square root of tensor with updated value and derivative
    """
    return tensor ** 0.5
