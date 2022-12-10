import numbers
from enum import Enum


_ALLOWED_TYPES = (int, float, numbers.Integral)
_SPECIAL_FUNCTIONS = "sqrt, exp, log, sin, cos, tan, arcsin, arccos, arctan"
_MAX_INDEPENDENT_VARS = 10
_GLOBAL_COUNTER = 0


class AdMode(Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class ArgMode(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
