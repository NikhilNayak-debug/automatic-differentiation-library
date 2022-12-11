import numbers
import numpy as np
from typing import Union, Iterable

from .fab_ad_tensor import FabTensor, AdMode
from .fab_ad_session import fab_ad_session


class AutoDiffOutput:
    def __init__(self, value: Union[numbers.Number, Iterable], gradient: Union[numbers.Number, Iterable]):
        self.value = value
        self.gradient = gradient

    def __str__(self) -> str:
        return f"Value: {self.value}\nGradient: {self.gradient}\n"


def auto_diff(output: Union[Iterable, FabTensor], mode=None) -> AutoDiffOutput:
    if mode == AdMode.FORWARD:
        return forward_mode_gradient(output)
    elif mode == AdMode.REVERSE:
        return reverse_mode_gradient(output)
    elif mode is None:
        n_input_nodes = fab_ad_session.global_tensor_count
        n_output_nodes = len(output) if type(output) is list else 1
        # TODO: add heuristic for optimal mode
        if n_input_nodes > n_output_nodes:
            return forward_mode_gradient(output)
        else:
            return reverse_mode_gradient(output)
    else:
        raise Exception(f"Invalid AD mode: {mode}!")


def forward_mode_gradient(output: Union[Iterable, FabTensor]) -> AutoDiffOutput:
    if isinstance(output, FabTensor):
        gradient = output.derivative[:fab_ad_session.global_tensor_count + 1]
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
            _gradient = tensor.derivative[:fab_ad_session.global_tensor_count + 1]
            if len(_gradient) == 1:
                _gradient = _gradient[0]
            value.append(tensor.value)
            gradient.append(_gradient)
        print(f"value: {value} gradient: {gradient}")
        return AutoDiffOutput(
            value=np.array(value),
            gradient=np.array(gradient)
        )
    else:
        raise TypeError(f"Gradient can be computed on either List of FabTensor or FabTensor, not object of type {type(output)}")


def reverse_mode_gradient_util(tensor, path_value=1):
    for source_tensor, local_gradient in tensor.source:
        new_path_value = path_value * local_gradient
        # print(f"Adding gradient {new_path_value} to tensor {source_tensor}")
        source_tensor.gradient += new_path_value
        reverse_mode_gradient_util(source_tensor, new_path_value)


def reverse_mode_gradient(output: Union[Iterable, FabTensor]) -> AutoDiffOutput:
    if isinstance(output, FabTensor):
        reverse_mode_gradient_util(output, path_value=1)
        return AutoDiffOutput(
            value=output.value,
            gradient=np.array([input_tensor.gradient for input_tensor in fab_ad_session.src_tensors]) if len(fab_ad_session.src_tensors) > 1 else fab_ad_session.src_tensors[0].gradient
        )
    elif isinstance(output, list):
        value = []
        gradient = []
        for output_tensor in output:
            reverse_mode_gradient_util(output_tensor, path_value=1)
            value.append(output_tensor.value)
            gradient.append(
                np.array([input_tensor.gradient for input_tensor in fab_ad_session.src_tensors]) if len(
                    fab_ad_session.src_tensors) > 1 else fab_ad_session.src_tensors[0].gradient
            )
        return AutoDiffOutput(
            value=value,
            gradient=gradient
        )
    else:
        raise TypeError(f"Gradient can be computed on either List of FabTensor or FabTensor, not object of type {type(output)}")

