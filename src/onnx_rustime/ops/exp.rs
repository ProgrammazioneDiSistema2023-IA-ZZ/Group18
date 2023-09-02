use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, stack_along_batch_dimension, tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use rayon::prelude::*;

/// `exp` - ONNX Node Implementation for Exponential Operation
///
/// Evaluates the exponential of each element within the input tensor. The computation takes place
/// concurrently for each batch present in the input tensor, optimizing performance, especially
/// for large input batches.
///
/// # Arguments
///
/// * `input` - A reference to the tensor consisting of values set to be exponentiated.
/// * `node` - A reference to the ONNX NodeProto which might have node-specific data required
///   during the subsequent conversion back to TensorProto.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor with computed exponentials, or
///   raises an error (`OnnxError`) if any phase of the operation experiences an issue.
///
/// # Errors
///
/// Potential errors include:
/// * Conversion from `TensorProto` to ndarray not succeeding.
///
/// # Example
///
/// ```rust
/// let exponential_output = exp(&input_tensor, &node);
/// ```
pub fn exp(input: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    // Convert the input TensorProto to an ndarray.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    // Extract batch size from the input tensor's shape.
    let batch_size = input_nd_array.shape()[0];

    // Compute the exponential for each batch in parallel.
    let results: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let batch_data = input_nd_array.index_axis(Axis(0), i);
            batch_data.mapv(|el| el.exp())
        })
        .collect();

    // Stack the results along the batch dimension.
    let stacked_result = stack_along_batch_dimension(results)?;

    // Convert the result ndarray back to TensorProto and return.
    convert_to_output_tensor(node, stacked_result)
}
