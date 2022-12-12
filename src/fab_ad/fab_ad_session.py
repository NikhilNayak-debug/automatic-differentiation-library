import numbers
import numpy as np
from typing import Iterable, Union

from .constants import _MAX_INDEPENDENT_VARS


class FabAdSession(object):

    def __init__(self, num_independent_tensors: int = _MAX_INDEPENDENT_VARS, global_tensor_count: int = -1) -> None:
        """init method

        Parameters
        ----------
        num_independent_tensors : int
            maximum number of seed vectors
        global_tensor_count : int
            current number of seed vectors
        """
        self.max_num_independent_tensors = num_independent_tensors
        self.global_tensor_count = global_tensor_count
        self.src_tensors = []
        self.dest_tensors = []
        self.all_tensors = []

    def get_index(self) -> int:
        """returns new index for independent variable

        Returns
        -------
        int
            returns new index for independent variable
        """
        self.global_tensor_count += 1
        if self.global_tensor_count >= self.max_num_independent_tensors:
            raise IndexError("Cannot compute gradient!")
        return self.global_tensor_count

    def initialize_derivative(self, value: Union[Iterable, ]) -> Iterable:
        """method for initializing derivative

        Parameters
        ----------
        value : Iterable

        Returns
        -------
        Iterable
            returns iterable for initialized derivative
        """
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
        """method for clearing independent variables and their derivatives
        """
        self.global_tensor_count = -1
        self.src_tensors = []
        self.all_tensors = []
        self.dest_tensors = []

    def initialize(self, num_inputs=_MAX_INDEPENDENT_VARS):
        self.clear()
        self.max_num_independent_tensors = num_inputs


fab_ad_session = FabAdSession()
