use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{NodeProto, TensorProto};
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, tensor_proto_to_ndarray,
};
use ndarray::*;

/// `concat` - ONNX Node Implementation for Concatenation
///
/// Concatenates a list of tensors along a specified axis into a single tensor.
/// All input tensors must have the same shape, except for the size of the specified axis.
///
/// # Arguments
///
/// * `inputs` - A reference to a vector containing the input tensors to concatenate.
/// * `node` - A reference to the ONNX NodeProto that describes the node in the ONNX computation graph.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Returns the concatenated output as a `TensorProto` or
///   an error (`OnnxError`) if the operation fails at any stage.
///
/// # Example
///
/// ```rust
/// let concatenated_result = concat(&input_tensors, &node);
/// ```
pub fn concat(inputs: &Vec<&TensorProto>, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    // Extract node attributes.
    let attributes = extract_attributes(node.get_attribute())?;
    let axis = get_int_attribute(&attributes, "axis", None)? as usize;

    let inputs_nd_array: Vec<_> = inputs
        .iter()
        .map(|tp| tensor_proto_to_ndarray::<f32>(tp).unwrap())
        .collect();

    let result = concat_tensors(inputs_nd_array, axis)?;

    convert_to_output_tensor(node, result)
}

fn concat_tensors<T, D>(
    tensors: Vec<ArrayBase<OwnedRepr<T>, D>>,
    axis: usize,
) -> Result<ArrayBase<OwnedRepr<T>, D>, OnnxError>
where
    T: Clone,
    D: Dimension + RemoveAxis,
{
    // Ensure all tensors have the same dimensionality
    let first_dim = tensors[0].ndim();

    if tensors.iter().any(|tensor| tensor.ndim() != first_dim) {
        return Err(OnnxError::ShapeError(
            "All tensors must have the same number of dimensions.".to_string(),
        ));
    }

    // Ensure the specified axis is valid
    if axis >= first_dim {
        return Err(OnnxError::ShapeError(
            "Specified axis is out of bounds for the given tensors.".to_string(),
        ));
    }

    // Convert to ArrayView for concatenate function
    let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();

    // Concatenate along the specified axis
    let concatenated_output = concatenate(Axis(axis), &views).map_err(|_| {
        OnnxError::ShapeError("Failed to concatenate tensors along specified axis.".to_string())
    })?;

    Ok(concatenated_output)
}
