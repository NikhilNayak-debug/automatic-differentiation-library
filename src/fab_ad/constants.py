import numbers
from enum import Enum


_ALLOWED_TYPES = (int, float, numbers.Integral)
_SPECIAL_FUNCTIONS = "sqrt, exp, log, sin, cos, tan, arcsin, arccos, arctan"


class AdMode(Enum):
    FORWARD = "forward"
    REVERSE = "reverse"
