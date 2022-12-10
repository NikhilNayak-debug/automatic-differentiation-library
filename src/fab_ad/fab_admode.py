import numpy as np
from fab_ad import FabTensor, fab_session
from constants import AdMode


class AutoDiffOutput:
    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

    def __str__(self):
        return f"Value: {self.value}\nGradient: {self.gradient}\n"


def auto_diff(output, mode):
    if mode == AdMode.FORWARD:
        return forward_mode_gradient(output)
    elif mode == AdMode.REVERSE:
        return reverse_mode_gradient(output)
    else:
        # TODO: add heuristic for optimal mode
        return None


def forward_mode_gradient(output):
    if isinstance(output, FabTensor):
        gradient = output.derivative[:fab_session.global_tensor_count + 1]
        if len(gradient) == 1:
            gradient = gradient[0]
        return AutoDiffOutput(
            value=output.value,
            gradient=gradient
        )
    elif isinstance(output, list):
        value = []
        gradient = []
        for tensor in output:
            assert isinstance(tensor, FabTensor)
            _gradient = tensor.derivative[:fab_session.global_tensor_count + 1]
            if len(_gradient) == 1:
                _gradient = _gradient[0]
            value.append(tensor.value)
            gradient.append(_gradient)
        return AutoDiffOutput(
            value=np.array(value),
            gradient=np.array(gradient)
        )
    else:
        raise TypeError(f"Gradient can be computed on either List of FabTensor or FabTensor, not object of type {type(output)}")
