use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, stack_along_batch_dimension,
    tensor_proto_to_ndarray,
};
use ndarray::prelude::*;

/// `reduce_sum` - ONNX Node Implementation for Reducing Sum Operation
///
/// The `reduce_sum` operation calculates the sum of the elements in the input tensor along
/// the given axes. Depending on the `keepdims` attribute, the resulting tensor can retain
/// the same rank as the input or might have the reduced dimension removed. Tensors with
/// rank zero are valid inputs.
///
/// This behavior closely mirrors the behavior of NumPy's sum operation, with the distinction
/// being that while NumPy defaults the `keepdims` parameter to `False`, this implementation
/// defaults it to `True`.
///
/// # Attributes
///
/// * `keepdims` : int (default is 1)
///   - Decides whether to keep the reduced dimension or not. A default value of 1 indicates
///     that the reduced dimension should be retained.
/// * `noop_with_empty_axes` : int (default is 0)
///   - Defines the behavior when the 'axes' attribute is empty. By default (`false`), all axes
///     are reduced. If the axes are empty and this attribute is set to true, the input tensor
///     won't be reduced, resulting in the output tensor being identical to the input tensor.
///
/// # Arguments
///
/// * `input` - A reference to the input tensor whose elements will be summed.
/// * `node` - A reference to the ONNX NodeProto containing node-specific data and attributes.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor after sum reduction. In the event
///   of an unsuccessful operation, an error (`OnnxError`) is returned.
///
/// # Errors
///
/// Potential errors include:
/// * Issues with extracting node attributes.
/// * Tensor conversion problems.
/// * Dimension mismatches during tensor operations.
///
/// # Example
///
/// ```rust
/// let reduced_tensor = reduce_sum(&input_tensor, &node);
/// ```
///
/// # Note
///
/// The operation will compute sums along the specified axis or if the axis is -1, it computes
/// the sum of all elements. If `noop_with_empty_axes` is set to 0 and the result has no elements,
/// the operation will return a copy of the input tensor.
pub fn reduce_sum(input: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let attributes = extract_attributes(node.get_attribute())?;

    let axis = get_int_attribute(&attributes, "axis", Some(-1))?;
    let keepdims = get_int_attribute(&attributes, "keepdims", Some(1))?;
    let noop = get_int_attribute(&attributes, "noop_with_empty_axes", Some(0))?;

    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    let batch_size = input_nd_array.shape()[0];

    let mut result_list = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let sample = input_nd_array.index_axis(Axis(0), b);

        let result = if axis == -1 {
            let sum = sample.sum();
            let sum_array: Array<f32, _> = Array::from_elem(IxDyn(&[1]), sum);
            sum_array.into_dyn()
        } else {
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (axis + sample.ndim() as i64) as usize
            };

            let reduced_array = sample.sum_axis(Axis(axis));

            if keepdims == 0 {
                let mut new_shape = reduced_array.shape().to_vec();
                new_shape[axis] = 1;
                reduced_array.into_shape(new_shape).unwrap()
            } else {
                reduced_array
            }
        };

        result_list.push(result);
    }

    // Stack the results together
    let result = stack_along_batch_dimension(result_list)?;

    let result = if noop == 0 && result.len() == 0 {
        input_nd_array.clone()
    } else {
        result
    };

    convert_to_output_tensor(node, result)
}
