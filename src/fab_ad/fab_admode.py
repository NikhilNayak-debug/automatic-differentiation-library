import numpy as np
from fab_ad import FabTensor, fab_session
from constants import AdMode


class AutoDiffOutput:
    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

    def __str__(self):
        return f"Value: {self.value}\nGradient: {self.gradient}\n"


def auto_diff(input=[], output=[], mode=None):
    if mode == AdMode.FORWARD:
        return forward_mode_gradient(output)
    elif mode == AdMode.REVERSE:
        return reverse_mode_gradient(input, output)
    elif mode is None:
        # TODO: add heuristic for optimal mode
        return None
    else:
        raise Exception()


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


def reverse_mode_gradient_util(tensor, path_value=1):
    for source_tensor, local_gradient in tensor.source:
        new_path_value = path_value * local_gradient
        print(f"Adding gradient {new_path_value} to tensor {source_tensor}")
        source_tensor.gradient += new_path_value
        reverse_mode_gradient_util(source_tensor, new_path_value)


def reverse_mode_gradient(input_tensor, output):
    if isinstance(output, FabTensor):
        reverse_mode_gradient_util(output, path_value=1)
        return AutoDiffOutput(
            value=output.value,
            gradient=input_tensor.gradient
        )
    elif isinstance(output, list):
        # TODO: implement multiple values
        raise Exception("Not yet implemented!")
    else:
        raise TypeError(f"Gradient can be computed on either List of FabTensor or FabTensor, not object of type {type(output)}")

