use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{convert_to_output_tensor, tensor_proto_to_ndarray};
use ndarray::prelude::*;

/// `add` - ONNX Node Implementation for Element-wise Addition
///
/// This function performs element-wise binary addition of tensors.
/// The operation supports broadcasting in the style of Numpy, allowing for
/// tensors of different shapes to be added together, given they are broadcast-compatible.
///
/// # Arguments
///
/// * `inputs` - A vector containing references to the input tensors.
/// * `initializers` - An optional vector containing references to the tensors
///   used as initializers or initializers. These tensors are added to the inputs
///   if provided.
/// * `node` - A reference to the ONNX NodeProto that describes the node in the
///   ONNX computation graph.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Returns the result of the addition as a `TensorProto` or
///   an error (`OnnxError`) if the operation fails at any stage.
///
/// # Errors
///
/// This function returns an error:
/// * If it fails to convert any TensorProto to an ndarray.
/// * If no tensors are provided for the addition operation.
///
/// # Example
///
/// ```rust
/// let result = add(&input_tensors, Some(&parameter_tensors), &node);
/// ```
///
pub fn add(
    inputs: &Vec<&TensorProto>,
    initializers: Option<&Vec<&TensorProto>>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    let inputs_nd_array = inputs
        .iter()
        .map(|x| {
            tensor_proto_to_ndarray::<f32>(x).map_err(|_| {
                OnnxError::ConversionError("Failed to convert TensorProto to ndarray".to_string())
            })
        })
        .collect::<Result<Vec<ArrayD<f32>>, OnnxError>>()?;

    let mut merged_tensors = Vec::new();
    merged_tensors.extend(inputs_nd_array);

    if let Some(param_tensors) = initializers {
        let initializers_nd_array = param_tensors
            .iter()
            .map(|x| {
                tensor_proto_to_ndarray::<f32>(x).map_err(|_| {
                    OnnxError::ConversionError(
                        "Failed to convert TensorProto to ndarray".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<ArrayD<f32>>, OnnxError>>()?;

        merged_tensors.extend(initializers_nd_array);
    }

    if merged_tensors.is_empty() {
        return Err(OnnxError::MissingInput(
            "No tensors provided for addition".to_string(),
        ));
    }

    // Element-wise addition of the tensors
    let result = merged_tensors
        .iter()
        .skip(1)
        .fold(merged_tensors[0].clone(), |acc, x| acc + x);

    convert_to_output_tensor(node, result)
}
