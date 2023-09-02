use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_float_attribute, stack_along_batch_dimension,
    tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use rayon::prelude::*;

/// `batch_normalization` - ONNX Node Implementation for Batch Normalization
///
/// Implements the batch normalization operation as described in the paper:
/// [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167).
///
/// Batch normalization operates in two modes:
/// * Inference mode (default): Uses the provided estimated statistics ('input_mean' and 'input_var').
/// * Training mode: Uses the running statistics.
///
/// # Arguments
///
/// * `input` - A reference to the input tensor 'X'.
/// * `initializers` - A vector containing references to the tensors 'scale', 'B', 'input_mean', and 'input_var'.
/// * `node` - A reference to the ONNX NodeProto that describes the node in the ONNX computation graph.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Returns the batch-normalized output as a `TensorProto` or
///   an error (`OnnxError`) if the operation fails at any stage.
///
/// # Example
///
/// ```rust
/// let result = batch_normalization(&input_tensor, &parameter_tensors, &node);
/// ```
pub fn batch_normalization(
    input: &TensorProto,
    initializers: &Vec<&TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Extract node attributes.
    let attributes = extract_attributes(node.get_attribute())?;
    let epsilon = get_float_attribute(&attributes, "epsilon", Some(1e-05))?;

    // Convert TensorProtos to ndarrays.
    let x = tensor_proto_to_ndarray::<f32>(input)?;
    let scale = tensor_proto_to_ndarray::<f32>(initializers[0])?;
    let bias = tensor_proto_to_ndarray::<f32>(initializers[1])?;
    let mean = tensor_proto_to_ndarray::<f32>(initializers[2])?;
    let var = tensor_proto_to_ndarray::<f32>(initializers[3])?;

    let batch_size = x.shape()[0];

    // Reshape tensors for broadcasting.
    let broadcasted_mean = mean
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast mean".into()))?;

    let broadcasted_var = var
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast variance".into()))?;

    let broadcasted_scale = scale
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast scale".into()))?;

    let broadcasted_bias = bias
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast bias".into()))?;

    // Process batches.
    let result_list: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let batch_data = x.index_axis(Axis(0), i);

            // Normalize using the formula: normalized = (x - mean) / sqrt(var + epsilon)
            let normalized =
                (&batch_data - &broadcasted_mean) / (&broadcasted_var + epsilon).mapv(|v| v.sqrt());

            // Apply scale and bias: out = normalized * scale + bias
            normalized * &broadcasted_scale + &broadcasted_bias
        })
        .collect();

    // Concatenate results along the batch dimension.
    let result = stack_along_batch_dimension(result_list)?;

    convert_to_output_tensor(node, result)
}
