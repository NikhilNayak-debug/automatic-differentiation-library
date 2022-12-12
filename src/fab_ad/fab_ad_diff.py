import numbers
import numpy as np
from typing import Union, Iterable

from .fab_ad_tensor import FabTensor, AdMode
from .fab_ad_session import fab_ad_session


class AutoDiffOutput:
    def __init__(self, value: Union[numbers.Number, Iterable], gradient: Union[numbers.Number, Iterable]):
        """init method

        Parameters
        ----------
        value : number
            intial function value
        gradient : array
            gradient w.r.t all seed vectors
        """
        self.value = value
        self.gradient = gradient

    def __str__(self) -> str:
        """Represents the AutoDiffOutput as a string

        Returns
        -------
        str
            AutoDiffOutput as a string
        """
        verbatim = ""
        if len(fab_ad_session.dest_tensors) > 1:
            for idx, tensor in enumerate(fab_ad_session.dest_tensors):
                if len(fab_ad_session.src_tensors) == 1:
                    gradient_str = "\n".join([f"Function {idx} Gradient w.r.t {src_tensor.identifier} = {self.gradient[idx]}" for src_tensor_id, src_tensor in enumerate(fab_ad_session.src_tensors)])
                else:
                    gradient_str = "\n".join([f"Function {idx} Gradient w.r.t {src_tensor.identifier} = {self.gradient[idx][src_tensor_id]}" for src_tensor_id, src_tensor in enumerate(fab_ad_session.src_tensors)])
                verbatim += f"Function {idx}: Value: {tensor.value}\n{gradient_str}\n"
        else:
            if len(fab_ad_session.src_tensors) == 1:
                gradient_str = "\n".join(
                    [f"Gradient w.r.t {src_tensor.identifier} = {self.gradient}" for src_tensor_id, src_tensor in
                     enumerate(fab_ad_session.src_tensors)])
            else:
                gradient_str = "\n".join(
                    [f"Gradient w.r.t {src_tensor.identifier} = {self.gradient[src_tensor_id]}" for
                     src_tensor_id, src_tensor in enumerate(fab_ad_session.src_tensors)])
            verbatim += f"Function 0: Value: {self.value}\n{gradient_str}\n"

        return verbatim


def auto_diff(output: Union[Iterable, FabTensor], mode=None) -> AutoDiffOutput:
    """returns gradient in either forward or reverse mode

        Parameters
        ----------
        output : FabTensor

        Returns
        -------
        number
            returns gradient in either forward or reverse mode
        
    """
    fab_ad_session.dest_tensors = []
    if mode == AdMode.FORWARD:
        result = forward_mode_gradient(output)
        return result
    elif mode == AdMode.REVERSE:
        result = reverse_mode_gradient(output)
        return result
    elif mode is None:
        n_input_nodes = fab_ad_session.global_tensor_count
        n_output_nodes = len(output) if type(output) is list else 1
        # TODO: improve heuristic for to identify mode
        if n_input_nodes > n_output_nodes:
            result = forward_mode_gradient(output)
            return result
        else:
            result = forward_mode_gradient(output)
            return result
    else:
        raise Exception(f"Invalid AD mode: {mode}!")


def forward_mode_gradient(output: Union[Iterable, FabTensor]) -> AutoDiffOutput:
    """returns forward_mode_gradient

        Parameters
        ----------
        output : FabTensor

        Returns
        -------
        number
            returns gradient in forward mode
        
    """
    if isinstance(output, FabTensor):
        fab_ad_session.dest_tensors.append(output)
        gradient = output.derivative[:fab_ad_session.global_tensor_count + 1]
        if len(gradient) == 1:
            gradient = gradient[0]
        return AutoDiffOutput(
            value=output.value,
            gradient=gradient,
        )
    elif isinstance(output, list):
        value = []
        gradient = []
        for tensor in output:
            assert isinstance(tensor, FabTensor)
            fab_ad_session.dest_tensors.append(tensor)
            _gradient = tensor.derivative[:fab_ad_session.global_tensor_count + 1]
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
    """util for reverse_mode_gradient

        Parameters
        ----------
        tensor : FabTensor
        path_value : gradient for specific path value

        Returns
        -------
        number
            returns gradient util in reverse mode
        
    """
    for source_tensor, local_gradient in tensor.source:
        new_path_value = path_value * local_gradient
        # print(f"Adding gradient {new_path_value} to tensor {source_tensor}")
        source_tensor.gradient += new_path_value
        reverse_mode_gradient_util(source_tensor, new_path_value)


def reverse_mode_gradient(output: Union[Iterable, FabTensor]) -> AutoDiffOutput:
    for tensor in fab_ad_session.all_tensors:
        tensor.zero_grad()
    """returns reverse_mode_gradient

        Parameters
        ----------
        output : FabTensor

        Returns
        -------
        number
            returns gradient in reverse mode
        
    """
    if isinstance(output, FabTensor):
        reverse_mode_gradient_util(output, path_value=1)
        fab_ad_session.dest_tensors.append(output)
        return AutoDiffOutput(
            value=output.value,
            gradient=np.array([input_tensor.gradient for input_tensor in fab_ad_session.src_tensors]) if len(fab_ad_session.src_tensors) > 1 else fab_ad_session.src_tensors[0].gradient
        )
    elif isinstance(output, list):
        value = []
        gradient = []
        for output_tensor in output:
            fab_ad_session.dest_tensors.append(output_tensor)
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

