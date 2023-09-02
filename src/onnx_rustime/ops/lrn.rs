use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_float_attribute, get_int_attribute,
    tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use rayon::prelude::*;

/// `lrn` - ONNX Node Implementation for Local Response Normalization (LRN) Operation
///
/// The LRN operation is used to perform normalization over local input regions and is
/// commonly found in convolutional neural networks. The local region is defined across the
/// channels for each element in a tensor. This operation is inspired from the AlexNet
/// paper. It provides a kind of "lateral inhibition" by normalizing over local input regions.
///
/// # Attributes
///
/// - `alpha` (float): Scaling parameter with a default value of `0.0001`.
/// - `beta` (float): Exponential factor with a default value of `0.75`.
/// - `bias` (float): Default value of `1.0`.
/// - `size` (int): Defines the number of channels to sum over, and it's a required attribute.
///
/// # Mathematical Explanation
///
/// Given a tensor element `X[n, c, d1, ..., dk]` of shape `(N x C x D1 x D2, ..., Dk)`, its
/// local region across the channels is computed as:
///
/// `{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.
///
/// The square sum for a region is:
///
/// `square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
/// where the values of i are within the defined local region.
///
/// The output tensor Y is calculated as:
///
/// `Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`.
///
/// # Arguments
///
/// * `input` - A reference to the tensor on which the LRN operation is to be applied.
/// * `node` - A reference to the ONNX NodeProto containing node-specific data and attributes.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor after performing the LRN operation.
///   In case of an unsuccessful operation, it returns an error (`OnnxError`).
///
/// # Errors
///
/// Possible errors include:
/// * Failed extraction of node attributes.
/// * Unsuccessful conversion from `TensorProto` to ndarray.
/// * LRN operation not being successful.
///
/// # Example
///
/// ```rust
/// let normalized_tensor = lrn(&input_tensor, &node);
/// ```
pub fn lrn(input: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    // Extract node attributes.
    let attributes = extract_attributes(node.get_attribute())?;
    let alpha: f32 = get_float_attribute(&attributes, "alpha", Some(0.0001))?;
    let beta: f32 = get_float_attribute(&attributes, "beta", Some(0.75))?;
    let bias: f32 = get_float_attribute(&attributes, "bias", Some(1.0))?;
    let size: usize = get_int_attribute(&attributes, "size", None)? as usize;

    // Convert TensorProto to ndarray.
    let x = tensor_proto_to_ndarray::<f32>(input)?;
    let shape = x.dim();
    let c = shape[1]; // Number of channels

    let mut square_sum = Array::zeros(shape);

    // Split square_sum into n slices along the batch dimension and iterate in parallel.
    square_sum
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(b, mut batch_slice)| {
            for idx in 0..c {
                let start = usize::max(
                    0,
                    idx.saturating_sub(((size - 1) as f32 / 2.0).floor() as usize),
                );
                let end = usize::min(c, idx + ((size - 1) as f32 / 2.0).ceil() as usize);

                for j in start..end {
                    let slice = x.slice(s![b, j, .., ..]);
                    batch_slice
                        .slice_mut(s![idx, .., ..])
                        .zip_mut_with(&slice.mapv(|v| v.powi(2)), |a, &b| *a += b);
                }
            }
        });

    let y = &x / (bias + alpha / (size as f32) * &square_sum).mapv(|v| v.powf(beta));

    convert_to_output_tensor(node, y)
}
