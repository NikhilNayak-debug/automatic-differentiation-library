from __future__ import annotations
import numbers
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enum import Enum
from typing import Iterable, Union

from .constants import _ALLOWED_NUMERICS
from .fab_ad_session import fab_ad_session


class AdMode(Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class FabTensor(object):

    def __init__(self, value: Union[Iterable, numbers.Number], derivative: Union[Iterable, numbers.Number] = None,
                 identifier: str = "", mode: Enum = AdMode.FORWARD, source: list = [], depth: int = 0):
        """init method

        Parameters
        ----------
        value : number
            evaluated function value
        derivative : array, optional
            derivative w.r.t all seed vectors, by default None
        identifier : str, optional
            function expression, by default ""
        """
        self.value = value
        if isinstance(self.value, Iterable):
            self.value = np.array(self.value)
        if derivative is None:
            # derivative w.r.t all independent variables
            derivative = fab_ad_session.initialize_derivative(value)
        self.depth = depth
        if self.depth == 0:
            # add tensor to list of source nodes in session
            fab_ad_session.src_tensors.append(self)
        fab_ad_session.all_tensors.append(self)
        if isinstance(derivative, (int, float, numbers.Integral, numbers.Number)):
            derivative = [derivative]
        self.derivative = np.array(derivative)
        self.identifier = identifier

        assert mode in [AdMode.FORWARD, AdMode.REVERSE]
        self.mode = mode
        self.source = source
        self._reverse_mode_gradient = 0

    def __repr__(self) -> str:
        """Represents the FabTensor as a string

        Returns
        -------
        str
            FabTensor as a string
        """
        return f"value: {self.value} derivative: {self.derivative}" \
               f" name: {self.identifier} reverse mode gradient: {self._reverse_mode_gradient}"

    def __str__(self) -> str:
        """Represents the FabTensor as a string

        Returns
        -------
        str
            FabTensor as a string
        """
        return f"value: {self.value} derivative: {self.derivative}" \
               f" name: {self.identifier} reverse mode gradient: {self._reverse_mode_gradient}"

    def __eq__(self, other) -> bool:
        """Checks if value attribute of two `FabTensor` objects are equal.

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        boolean
            if value attribute of two `FabTensor` objects are equal
        """
        if isinstance(other, FabTensor):
            return self.value == other.value
        elif isinstance(other, _ALLOWED_NUMERICS):
            return self.value == other
        else:
            raise TypeError(f"Cannot compare FabTensor and object of type {type(other)}")

    def __ne__(self, other) -> bool:
        """Checks if value attribute of two `FabTensor` objects are not equal.

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        boolean
            if value attribute of two `FabTensor` objects are not equal
        
        """
        if isinstance(other, FabTensor):
            return self.value != other.value
        elif isinstance(other, _ALLOWED_NUMERICS):
            return self.value != other
        else:
            raise TypeError(f"Cannot compare FabTensor and object of type {type(other)}")

    def __lt__(self, other) -> bool:
        """Checks if value attribute of self is less than other `FabTensor` object

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        boolean
            if value attribute of self is less than other `FabTensor` object
        
        """
        if isinstance(other, FabTensor):
            return self.value < other.value
        elif isinstance(other, _ALLOWED_NUMERICS):
            return self.value < other
        else:
            raise TypeError(f"Cannot compare FabTensor and object of type {type(other)}")

    def __gt__(self, other) -> bool:
        """Checks if value attribute of self is greater than other `FabTensor` object

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        boolean
            if value attribute of self is greater than other `FabTensor` object
        
        """
        if isinstance(other, FabTensor):
            return self.value > other.value
        elif isinstance(other, _ALLOWED_NUMERICS):
            return self.value > other
        else:
            raise TypeError(f"Cannot compare FabTensor and object of type {type(other)}")

    def __le__(self, other) -> bool:
        """Checks if value attribute of self is less than or equal to other `FabTensor` object

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        boolean
            if value attribute of self is less than or equal to other `FabTensor` object
        
        """
        if isinstance(other, FabTensor):
            return self.value <= other.value
        elif isinstance(other, _ALLOWED_NUMERICS):
            return self.value <= other
        else:
            raise TypeError(f"Cannot compare FabTensor and object of type {type(other)}")

    def __ge__(self, other) -> bool:
        """Checks if value attribute of self is greater than or equal to other `FabTensor` object

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        boolean
            if value attribute of self is greater than or equal to other `FabTensor` object
        
        """
        if isinstance(other, FabTensor):
            return self.value >= other.value
        elif isinstance(other, _ALLOWED_NUMERICS):
            return self.value >= other
        else:
            raise TypeError(f"Cannot compare FabTensor and object of type {type(other)}")

    def __len__(self) -> int:
        """return length of derivative array

        Returns
        -------
        array
            length of derivative array
        
        """
        if self.derivative is not None:
            return len(self.derivative)
        else:
            raise ValueError("derivative is not initialized yet!")
    
    def __neg__(self) -> FabTensor:
        """negation of `FabTensor` object

        Returns
        -------
        FabTensor
            negation of `FabTensor` object
        """
        return -1 * self

    def __add__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """sum of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            sum of two `FabTensor` objects
        
        """
        if isinstance(other, FabTensor):
            return FabTensor(
                self.value + other.value,
                derivative=self.derivative + other.derivative,
                identifier=f'{self.identifier} + {other.identifier}',
                mode=self.mode,
                source=[
                    (self, 1),
                    (other, 1)
                ], depth=self.depth + 1)
        elif isinstance(other, _ALLOWED_NUMERICS):
            return FabTensor(
                self.value + other,
                derivative=self.derivative,
                identifier=f'{self.identifier} + {other}',
                mode=self.mode,
                source=[
                    (self, 1),
                ], depth=self.depth + 1)
        else:
            raise TypeError(f"addition not supported between types FabTensor and {type(other)}")

    def __radd__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """sum of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            sum of two `FabTensor` objects
        
        """
        if isinstance(other, FabTensor):
            return FabTensor(
                self.value + other.value,
                derivative=self.derivative + other.derivative,
                identifier=f'{other.identifier} + {self.identifier}',
                mode=self.mode,
                source=[
                    (self, 1),
                    (other, 1),
                ], depth=self.depth + 1)
        elif isinstance(other, _ALLOWED_NUMERICS):
            return FabTensor(
                self.value + other,
                derivative=self.derivative,
                identifier=f'{other} + {self.identifier}',
                mode=self.mode,
                source=[
                    (self, 1),
                ], depth=self.depth + 1)
        else:
            raise TypeError(f"addition not supported between types FabTensor and {type(other)}")

    def __iadd__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """sum of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            sum of two `FabTensor` objects
        """
        return self + other
    
    def __sub__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """difference of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            difference of two `FabTensor` objects
        
        """
        if isinstance(other, FabTensor):
            return FabTensor(
                self.value - other.value,
                derivative=self.derivative - other.derivative,
                identifier=f'{self.identifier} - {other.identifier}',
                mode=self.mode,
                source=[
                    (self, 1),
                    (other, -1)
                ], depth=self.depth + 1)
        elif isinstance(other, _ALLOWED_NUMERICS):
            return FabTensor(
                self.value - other,
                derivative=self.derivative,
                identifier=f'{self.identifier} - {other}',
                mode=self.mode,
                source=[
                    (self, 1)
                ], depth=self.depth + 1)
        else:
            raise TypeError(f"addition not supported between types FabTensor and {type(other)}")
    
    def __rsub__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """difference of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            difference of two `FabTensor` objects
        """
        if isinstance(other, FabTensor):
            return FabTensor(
                other.value - self.value,
                derivative=other.derivative - self.derivative,
                identifier=f'{other.identifier} - {self.identifier}',
                mode=self.mode,
                source=[
                    (other, 1),
                    (self, -1),
                ], depth=self.depth + 1)
        elif isinstance(other, _ALLOWED_NUMERICS):
            return FabTensor(
                other - self.value,
                derivative=-1 * self.derivative,
                identifier=f'{other} - {self.identifier}',
                mode=self.mode,
                source=[
                    (self, -1),
                ], depth=self.depth + 1)
        else:
            raise TypeError(f"addition not supported between types {type(other)} and FabTensor")
    
    def __isub__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """difference of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            difference of two `FabTensor` objects
        """
        return self - other
    
    def __mul__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """product of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            product of two `FabTensor` objects
        
        """
        if isinstance(other, FabTensor):
            return FabTensor(
                self.value * other.value,
                derivative=self.value * other.derivative + other.value * self.derivative,
                identifier=f'{self.identifier} * {other.identifier}',
                mode=self.mode,
                source=[
                    (self, other.value),
                    (other, self.value),
                ], depth=self.depth + 1)
        elif isinstance(other, _ALLOWED_NUMERICS):
            if other == 1:
                identifier = self.identifier
            elif other == -1:
                identifier=f'-{self.identifier}'
            else:
                identifier = f'{self.identifier} * {other}'
            return FabTensor(
                self.value * other,
                derivative=self.derivative * other,
                identifier=identifier,
                mode=self.mode,
                source=[
                    (self, other),
                ], depth=self.depth + 1)
        else:
            raise TypeError(f"Cannot multiple FabTensor with object of type {type(other)}")

    def __rmul__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """product of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            product of two `FabTensor` objects
        """
        if isinstance(other, FabTensor):
            return FabTensor(
                self.value * other.value,
                derivative=self.value * other.derivative + other.value * self.derivative,
                identifier=f'{other.identifier} * {self.identifier}',
                mode=self.mode,
                source=[
                    (self, other.value),
                    (other, self.value),
                ], depth=self.depth + 1)
        elif isinstance(other, _ALLOWED_NUMERICS):
            if other == 1:
                identifier = self.identifier
            elif other == -1:
                identifier=f'-{self.identifier}'
            else:
                identifier = f'{other} * {self.identifier}'
            return FabTensor(
                self.value * other,
                derivative=self.derivative * other,
                identifier=identifier,
                mode=self.mode,
                source=[
                    (self, other),
                ], depth=self.depth + 1)
        else:
            raise TypeError(f"Cannot multiple FabTensor with object of type {type(other)}")

    def __imul__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """product of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            product of two `FabTensor` objects
        """
        return self * other
    
    def __truediv__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """division of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            division of two `FabTensor` objects
        """
        try:
            return self * (other ** (-1))
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Cannot divide FabTensor with {other}")
        except Exception as e:
            raise e
    
    def __rtruediv__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """division of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            division of two `FabTensor` objects
        """
        return (self ** -1) * other
    
    def __itruediv__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """division of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            division of two `FabTensor` objects
        """
        return self * (other ** (-1))

    def __pow__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """power of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            power of two `FabTensor` objects
        """
        if isinstance(other, FabTensor):
            value = self.value ** other.value
            derivative = (other.value * (self.value ** (other.value - 1)) * self.derivative) + ((self.value ** other.value) * np.log(self.value) * other.derivative)
            return FabTensor(
                value=value,
                derivative=derivative,
                identifier=f"{self.identifier}^{other.identifier}",
                mode=self.mode,
                source=[
                    (self, other.value * (self.value ** (other.value - 1))),
                    (other, (self.value ** other.value) * np.log(self.value))
                ], depth=self.depth + 1
            )
        elif isinstance(other, _ALLOWED_NUMERICS):
            return FabTensor(
                value=self.value ** other,
                derivative=other * (self.value ** (other - 1)) * self.derivative,
                identifier=f"{self.identifier}^{other}" if other != -1 else f"1 / {self.identifier}",
                mode=self.mode,
                source=[
                    (self, other * (self.value ** (other - 1)))
                ], depth=self.depth + 1
            )
        else:
            raise TypeError(f"Cannot compute power of FabTensor with object of type {type(other)}")

    def __rpow__(self, other: Union[numbers.Number, FabTensor]) -> FabTensor:
        """power of two `FabTensor` objects

        Parameters
        ----------
        other : FabTensor

        Returns
        -------
        FabTensor
            power of two `FabTensor` objects
        """
        if isinstance(other, _ALLOWED_NUMERICS):
            return FabTensor(
                value=other ** self.value,
                derivative=(other ** self.value) * np.log(other) * self.derivative,
                identifier=f"{other}^{self.identifier}",
                mode=self.mode,
                source=[
                    (self, (other ** self.value) * np.log(other))
                ], depth=self.depth + 1,
            )
        else:
            raise TypeError(f"Cannot compute power of object of type {type(other)} with FabTensor")

    def directional_derivative(self, seed_vector: Union[np.ndarray, Iterable]) -> numbers.Number:
        """directional derivative w.r.t alls seed vectors

        Parameters
        ----------
        seed_vector : np.array
            array of seed vectors or all the independent variables

        Returns
        -------
        number
            directional derivative w.r.t given seed vectors
        """
        return np.array(seed_vector).dot(self.derivative)


    @property
    def gradient(self) -> numbers.Number:
        """returns reverse mode gradient

        Returns
        -------
        number
            returns reverse mode gradient
        """
        return self._reverse_mode_gradient


    @gradient.setter
    def gradient(self, value) -> None:
        """setting reverse mode gradient

        Parameters
        ----------
        value : np.array or number
            gradient as array or number
        """
        self._reverse_mode_gradient = value

    def zero_grad(self) -> None:
        """setting reverse mode gradient to zero
        """
        self._reverse_mode_gradient = 0
