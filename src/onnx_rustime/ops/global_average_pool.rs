use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{NodeProto, TensorProto};
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, stack_along_batch_dimension,
    tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use rayon::prelude::*;

/// `global_average_pool` - ONNX Node Implementation for Global Average Pooling Operation
///
/// Computes the average value for each channel across all spatial dimensions, which
/// effectively condenses the spatial dimensions into a single averaged value. The resultant
/// tensor shape is `[batch_size, channels, 1, 1]`.
///
/// # Arguments
///
/// * `inputs` - A reference to the tensor set for global average pooling.
/// * `node` - A reference to the ONNX NodeProto containing node-specific data and
///   potential attributes.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor after performing the global
///   average pooling operation. In case of an unsuccessful operation, it returns an error
///   (`OnnxError`).
///
/// # Errors
///
/// Possible errors include:
/// * Failed extraction of node attributes.
/// * Unsuccessful conversion from `TensorProto` to ndarray.
/// * Global average pooling operation not being successful.
///
pub fn global_average_pool(
    inputs: &TensorProto,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Extract node attributes (not used in this function, but might be needed for future extensions).
    let _attributes = extract_attributes(node.get_attribute())?;

    // Convert the input TensorProto to an ndarray.
    let inputs_nd_array = tensor_proto_to_ndarray::<f32>(inputs)?;

    // Perform global average pooling on the ndarray.
    let result = global_average_pooling(&inputs_nd_array)?;

    // Convert the result ndarray back to TensorProto and return.
    convert_to_output_tensor(node, result)
}

/// Performs global average pooling on the given tensor.
///
/// This helper function computes the global average for each channel in the
/// input tensor. It expects the input tensor shape to be `[batch_size, channels, ..., ...]`.
/// The output tensor will have a shape of `[batch_size, channels, 1, 1]`.
///
/// # Arguments
/// - `input_tensor`: The input tensor for which the global average pooling is computed.
///
/// # Returns
/// - A `Result` containing the tensor after performing global average pooling or
///   an error of type `OnnxError` if the pooling operation fails.
///
fn global_average_pooling(input_tensor: &ArrayD<f32>) -> Result<ArrayD<f32>, OnnxError> {
    // Get the batch size and number of channels from the input tensor shape.
    let batch_size = input_tensor.shape()[0];
    let channels = input_tensor.shape()[1];

    // Perform global average pooling for each batch and channel.
    let pooled_results: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|b| {
            let mut channel_averages = vec![0.0; channels];
            for c in 0..channels {
                let channel_data = input_tensor
                    .index_axis(Axis(0), b)
                    .index_axis(Axis(0), c)
                    .to_owned();

                // Calculate the average value for the current channel.
                let average_value = channel_data.mean().unwrap();
                channel_averages[c] = average_value;
            }
            // Convert channel averages into a tensor of shape [channels, 1, 1].
            ArrayD::from_shape_vec(IxDyn(&[channels, 1, 1]), channel_averages)
                .expect("Failed to create tensor from averages")
        })
        .collect();

    // Stack the results along the batch dimension to produce the final output tensor.
    stack_along_batch_dimension(pooled_results)
}
