# This Python stub provides type hints and documentation that reflects the functionality of the Rust backend.
# The comments explain that these functions are essentially wrappers around Rust functions,
# and the use of IDs is highlighted as a mechanism to abstract away the internal Rust data structures.
#
# This should help developers understand the function signatures and what each function does,
# even if the actual Rust implementations are hidden behind the scenes.
# When using tools like IntelliSense in VS Code, these docstrings will provide valuable context and guidance.

from typing import Tuple, Union

class PyException(Exception):
    """Custom exception for errors during Rust function execution."""

    pass

class PyValueError(ValueError):
    """Exception raised for value errors, e.g., invalid model or data IDs."""

    pass

def py_load_model(path: str) -> int:
    """
    Load an ONNX model from the provided path.

    Internally, this function uses the OnnxParser to load a ModelProto from
    the given path. The ModelProto is then stored and its corresponding ID
    (acting as an opaque pointer) is returned.

    Args:
    - path (str): The path to the ONNX model.

    Returns:
    - int: An ID corresponding to the loaded ModelProto.
    """
    ...

def py_load_data(path: str) -> int:
    """
    Load ONNX data (TensorProto) from the provided path.

    This function reads a TensorProto from the given path and stores it.
    It then returns an ID corresponding to this TensorProto, which acts
    as an opaque pointer.

    Args:
    - path (str): The path to the ONNX data.

    Returns:
    - int: An ID corresponding to the loaded TensorProto.
    """
    ...

def py_print_data(data_id: int) -> None:
    """
    Print the ONNX data (TensorProto) associated with the provided ID.

    The function retrieves the TensorProto using the ID, converts it to
    an ndarray and then prints it.

    Args:
    - data_id (int): The ID of the TensorProto to print.
    """
    ...

def py_run(model_id: int, input_data_id: int, verbose: bool) -> int:
    """
    Run the ONNX model with the provided input data.

    This function retrieves the ModelProto and TensorProto using their
    respective IDs, then executes the model with the input data. The
    resulting output TensorProto is stored and its ID is returned.

    Args:
    - model_id (int): The ID of the ModelProto (ONNX model) to run.
    - input_data_id (int): The ID of the input TensorProto.
    - verbose (bool): Whether to print verbose output during execution.

    Returns:
    - int: An ID corresponding to the output TensorProto.
    """
    ...

def py_display_outputs(predicted_data_id: int, expected_data_id: int) -> None:
    """
    Display the predicted and expected outputs.

    This function retrieves the predicted and expected TensorProtos using
    their IDs and then displays their contents in a pretty and fancy format.

    Args:
    - predicted_data_id (int): The ID of the predicted TensorProto.
    - expected_data_id (int): The ID of the expected TensorProto.
    """
    ...
