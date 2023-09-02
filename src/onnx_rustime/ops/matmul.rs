use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{NodeProto, TensorProto};
use crate::onnx_rustime::ops::gemm::gemm;

/// `matmul` - ONNX Node Implementation for Matrix Multiplication (MatMul) Operation
///
/// The `matmul` operation is used to perform matrix multiplication, behaving similarly to
/// numpy's `matmul` function. It's an operation that multiplies matrix-like arrays in a
/// manner consistent with the semantics of the numpy.matmul function. This is different
/// from dot product as it supports broadcasting and has specific behavior for 1-D arrays.
///
/// The operation multiplies two tensors and produces a tensor that is the matrix product
/// of the two input tensors. Refer to numpy's documentation for further clarity:
/// [numpy.matmul](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html)
///
/// Internally, this function leverages the `gemm` (General Matrix Multiply) operation,
/// which is a more general case for matrix multiplication. `gemm` handles possible biases,
/// scalar multipliers, and transposition, whereas `matmul` provides a simpler interface for
/// straightforward matrix multiplication.
///
/// # Arguments
///
/// * `inputs` - A vector reference containing the two tensors to be multiplied.
/// * `initializers` - An optional vector reference containing additional tensor initializers,
///   if any. In the current context, these initializers are directly forwarded to the `gemm` function.
/// * `node` - A reference to the ONNX NodeProto containing node-specific data and attributes.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the resulting tensor after matrix multiplication.
///   In case of an unsuccessful operation, it returns an error (`OnnxError`).
///
/// # Errors
///
/// Possible errors include any that might arise from the underlying `gemm` function, such as:
/// * Failed extraction of node attributes.
/// * Dimension mismatches during matrix multiplication.
/// * Other multiplication-specific errors.
///
/// # Example
///
/// ```rust
/// let result_tensor = matmul(&input_tensors, Some(&initializers), &node);
/// ```
///
/// # Note
///
/// While `matmul` provides a simple interface tailored for matrix multiplication, the
/// underlying mechanics are powered by the more robust `gemm` function. This design ensures
/// optimization and support for more advanced features without complicating the basic matrix
/// multiplication use case.
pub fn matmul(
    inputs: &Vec<&TensorProto>,
    initializers: Option<&Vec<&TensorProto>>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    gemm(inputs, initializers, node)
}
