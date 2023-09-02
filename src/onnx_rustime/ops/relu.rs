use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{NodeProto, TensorProto};
use crate::onnx_rustime::ops::utils::tensor_proto_to_ndarray;
use ndarray::ArrayD;

use super::utils::convert_to_output_tensor;

/// `relu` - ONNX Node Implementation for Rectified Linear Unit (ReLU) Activation
///
/// The `relu` operation applies the Rectified Linear Unit (ReLU) activation function to the input tensor.
/// The ReLU function is defined as y = max(0, x) and is applied elementwise to the input tensor.
/// Consequently, any negative values in the tensor are set to 0, while non-negative values remain unchanged.
///
/// # Arguments
///
/// * `input` - A reference to the input tensor to which the ReLU activation will be applied.
/// * `node` - A reference to the ONNX NodeProto containing node-specific data. Note that while
///   this argument is provided, the ReLU operation does not typically make use of any node-specific attributes.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor after applying the ReLU activation.
///   In case of any issues, an error (`OnnxError`) will be returned.
///
/// # Errors
///
/// Potential errors include:
/// * Unsuccessful conversion from `TensorProto` to ndarray.
/// * Mismatches in tensor shapes during reshaping operations.
///
/// # Example
///
/// ```rust
/// let relu_activated_tensor = relu(&input_tensor, &node);
/// ```
pub fn relu(input: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    let relu_values: Vec<f32> = input_nd_array
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect();

    let result = ArrayD::from_shape_vec(input_nd_array.raw_dim(), relu_values)
        .map_err(|_| OnnxError::ShapeMismatch("Failed to reshape!".into()))?;

    convert_to_output_tensor(node, result)
}
