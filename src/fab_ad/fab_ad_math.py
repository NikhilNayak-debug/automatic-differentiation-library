import math
import numbers
import os
import sys
from typing import Union
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from .fab_ad_tensor import FabTensor
from .constants import _ALLOWED_NUMERICS, _SPECIAL_FUNCTIONS


def sin(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1)
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=np.sin(tensor), derivative = 0, identifier="sin(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def cos(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1)
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=np.cos(tensor), derivative=0, identifier="cos(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def tan(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1)
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=np.tan(tensor), derivative=0, identifier="tan(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def cosec(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
    """cosec of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cosec of tensor with updated value and derivative
    """
    return 1 / sin(tensor)


def sec(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
    """sec of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        sec of tensor with updated value and derivative
    """
    return 1 / cos(tensor)


def cot(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
    """cot of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cot of tensor with updated value and derivative
    """
    return 1 / tan(tensor)


def arcsin(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1
        )
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        if not (-1 <= tensor <= 1):
            raise ValueError("Value of tensor out of range for function arcsin!")
        return FabTensor(value=np.arcsin(tensor), derivative = 0, identifier="sin^{-1}(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def arccos(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1
        )
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        if not (-1 <= tensor <= 1):
            raise ValueError("Value of tensor out of range for function arccos!")
        return FabTensor(value=np.arccos(tensor), derivative = 0, identifier="cos^{-1}(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def arctan(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1
        )
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=np.arctan(tensor), derivative = 0, identifier="tan^{-1}(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def arccosec(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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


def arcsec(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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


def arccot(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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


def exp(tensor: Union[FabTensor, numbers.Number, np.ndarray], base: numbers.Number = np.e) -> FabTensor:
    """exponential of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor
    base: float

    Returns
    -------
    FabTensor
        exponential of tensor with updated value and derivative

    """
    return base ** tensor


def sinh(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1
        )
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=np.sinh(tensor), derivative=0, identifier="sinh(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def cosh(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
            ], depth=tensor.depth + 1
        )
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=np.cosh(tensor), derivative=0, identifier="cosh(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def tanh(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
    """tanh of tensor with updated value and derivative

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
            ], depth=tensor.depth + 1
        )
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=np.cosh(tensor), derivative=0, identifier="tanh(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def cosech(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
    """cosech of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cosh of tensor with updated value and derivative
    """
    return 1 / sinh(tensor)


def sech(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
    """sech of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cosh of tensor with updated value and derivative
    """
    return 1 / cosh(tensor)


def coth(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
    """coth of tensor with updated value and derivative

    Parameters
    ----------
    tensor : FabTensor

    Returns
    -------
    FabTensor
        cosh of tensor with updated value and derivative
    """
    return 1 / tanh(tensor)


def logistic(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        return FabTensor(value=1 / (1 + np.exp(-tensor)), derivative=0, identifier=f"logistic({tensor})")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def log(tensor: Union[FabTensor, numbers.Number, np.ndarray], base: numbers.Number = np.e) -> FabTensor:
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
            ], depth=tensor.depth + 1
        )
    elif isinstance(tensor, _ALLOWED_NUMERICS):
        if tensor < 0.0:
            raise ValueError("Value of tensor out of range for function log!")
        return FabTensor(value=np.log(tensor), derivative=0, identifier="log(input)")
    else:
        raise TypeError(f"Methods {_SPECIAL_FUNCTIONS} can be used on FabTensor objects and {_ALLOWED_NUMERICS} only!")


def sqrt(tensor: Union[FabTensor, numbers.Number, np.ndarray]) -> FabTensor:
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
