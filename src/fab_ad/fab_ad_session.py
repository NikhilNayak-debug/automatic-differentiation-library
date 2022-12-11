import numbers
import numpy as np
from typing import Iterable, Union, Type

from .constants import _MAX_INDEPENDENT_VARS


class FabAdSession(object):

    def __init__(self, num_independent_tensors: int = _MAX_INDEPENDENT_VARS, global_tensor_count: int = -1) -> None:
        self.max_num_independent_tensors = num_independent_tensors
        self.global_tensor_count = global_tensor_count
        self.src_tensors = []

    def get_index(self) -> int:
        self.global_tensor_count += 1
        if self.global_tensor_count >= self.max_num_independent_tensors:
            raise IndexError("Cannot compute gradient!")
        return self.global_tensor_count

    def initialize_derivative(self, value: Union[Iterable, ]) -> Iterable:
        if isinstance(value, numbers.Number):
            derivative = np.zeros(self.max_num_independent_tensors)
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            m = len(value)
            derivative = np.zeros((self.max_num_independent_tensors, m))
        else:
            raise TypeError(f"Invalid value of type {type(value)}!")
        index = self.get_index()
        derivative[index] = 1
        return derivative

    def clear(self) -> None:
        self.global_tensor_count = -1
        self.src_tensors = []


fab_ad_session = FabAdSession()
