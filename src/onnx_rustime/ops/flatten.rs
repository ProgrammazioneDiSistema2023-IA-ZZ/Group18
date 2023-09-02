use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{convert_to_output_tensor, tensor_proto_to_ndarray};

/// `flatten` - ONNX Node Implementation for Flatten Operation
///
/// Reshapes the provided input tensor into a 2D matrix by flattening the tensor's
/// dimensions, starting from the specified `axis` all the way to its last dimension.
/// When `axis` is not supplied, it uses the default value of 1.
///
/// As an illustration, for an input tensor of shape `[a, b, c, d]` combined with an
/// `axis` value of 2, the output tensor shape would be `[a * b, c * d]`.
///
/// # Arguments
///
/// * `input` - A reference to the tensor set for flattening.
/// * `node` - A reference to the ONNX NodeProto that may possess node-specific
///   attributes. It particularly looks for the `axis` attribute which ascertains
///   the starting dimension for the flatten operation.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor in its flattened form, or
///   returns an error (`OnnxError`) if any part of the operation is unsuccessful.
///
/// # Errors
///
/// Possible error scenarios are:
/// * Conversion from `TensorProto` to ndarray not being successful.
/// * Reshape operation based on the calculated output shape not being successful.
///
/// # Example
///
/// ```rust
/// let flattened_tensor = flatten(&input_tensor, &node);
/// ```
pub fn flatten(input: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    // Extract dimensions from the input tensor.
    let input_shape = input.get_dims();
    let input_first = input_shape[0] as usize;

    // Convert the input TensorProto to an ndarray.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    let mut output_shape: Vec<usize> = Vec::new();

    // Extract the 'axis' attribute from the node.
    let axis_attribute = node
        .get_attribute()
        .iter()
        .find(|attr| attr.get_name() == "axis");

    // Determine the axis value; default is 1 if not provided.
    let axis = axis_attribute.map_or(1, |attr| attr.get_i() as usize);

    // Compute the total number of elements from the axis to the last dimension.
    let total_elements = input_shape.clone()[1..].iter().product::<i64>() as usize;

    // Determine the shape of the output tensor based on the axis.
    if axis <= 1 {
        output_shape = vec![input_first, total_elements];
    } else {
        let mut outer_dim = 1;
        let mut inner_dim = 1;

        for (index, &dim) in input_shape.iter().enumerate() {
            if index < axis {
                outer_dim *= dim as usize;
            } else {
                inner_dim *= dim as usize;
            }
        }

        output_shape.push(outer_dim);
        output_shape.push(inner_dim);
    }

    // Reshape the input ndarray to the output shape.
    let result = input_nd_array.into_shape(output_shape).unwrap();

    // For debugging: print the shape of the result.
    println!("shape {:?}", result.shape());

    // Convert the reshaped ndarray back to TensorProto and return.
    convert_to_output_tensor(node, result)
}
