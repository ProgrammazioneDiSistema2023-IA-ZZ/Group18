use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{NodeProto, TensorProto};
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, get_ints_attribute,
    pad_matrix_2d, tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use rayon::prelude::*;

/// `maxpool` - ONNX Node Implementation for Maximum Pooling Operation
///
/// The `maxpool` operation is utilized to perform maximum pooling across the input tensor
/// based on provided kernel sizes, stride sizes, and padding lengths. Maximum pooling involves
/// selecting the maximum value from a subset of the input tensor according to the specified kernel
/// size and then downsampling the data into the output tensor for subsequent processing. This operation
/// is commonly used in Convolutional Neural Networks (CNNs) to reduce spatial dimensions and introduce
/// spatial invariances.
///
/// The calculation of the output spatial shape differs based on whether explicit padding (defined by `pads`)
/// or auto padding (defined by the now DEPRECATED `auto_pad` attribute) is employed. The operation's behavior
/// can also be influenced by the `ceil_mode` attribute.
///
/// Detailed computation equations and descriptions can be found in the official documentation:
/// [MaxPool Official ONNX Docs](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool).
///
/// # Arguments
///
/// * `inputs` - A reference to the input tensor to be max pooled.
/// * `node` - A reference to the ONNX NodeProto containing node-specific data and attributes.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor after max pooling. In case of
///   an unsuccessful operation, it returns an error (`OnnxError`).
///
/// # Errors
///
/// Possible errors include, but are not limited to:
/// * Failure in extracting node attributes.
/// * Mismatch in tensor dimensions for pooling.
/// * Inconsistencies in kernel and stride shapes.
///
/// # Example
///
/// ```rust
/// let result_tensor = maxpool(&input_tensor, &node);
/// ```
///
/// # Note
///
/// The internal behavior of the function (such as calculations for output shape) is
/// largely determined by various attributes like `kernel_shape`, `pads`, and `strides`.
/// Additional attributes like `ceil_mode` and `storage_order` are extracted but not currently
/// utilized. While `auto_pad` is mentioned in the official documentation, it is marked as
/// DEPRECATED, and its usage should be avoided in modern implementations.
pub fn maxpool(inputs: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let attributes = extract_attributes(node.get_attribute())?;

    // 2D-maxpool kernel and stride are always 2D
    let kernel_shape = get_ints_attribute(&attributes, "kernel_shape", Some(vec![1, 1]))?;
    let pads = get_ints_attribute(&attributes, "pads", Some(vec![0, 0, 0, 0]))?;
    let strides = get_ints_attribute(&attributes, "strides", Some(vec![1, 1]))?;

    // TODO: ceil_mode storage_order and dilations are not used
    let _ceil_mode = get_int_attribute(&attributes, "ceil_mode", Some(0))?;
    let _storage_order = get_int_attribute(&attributes, "storage_order", Some(0))?;
    let _dilations = get_ints_attribute(&attributes, "dilations", Some(vec![1, 1]))?;

    let inputs_nd_array = tensor_proto_to_ndarray::<f32>(inputs)?;

    let result = pool(&inputs_nd_array, &kernel_shape, &pads, &strides)?;

    convert_to_output_tensor(node, result)
}

fn pool(
    input_matrix: &ArrayD<f32>,
    kernel_shape: &Vec<i64>,
    pads: &Vec<i64>,
    strides: &Vec<i64>,
) -> Result<ArrayD<f32>, OnnxError> {
    // Choose the indices of dimensions to extract
    let batch_size = input_matrix.shape()[0];
    let channels = input_matrix.shape()[1];

    // Extract kernel shape and strides
    let kernel_height = kernel_shape[0] as usize;
    let kernel_width = kernel_shape[1] as usize;
    let stride_height = strides[0] as usize;
    let stride_width = strides[1] as usize;

    let mut pooled_results = Vec::new();

    for b in 0..batch_size {
        for c in 0..channels {
            // Extract the height-width input matrix
            let h_w_matrix = input_matrix
                .index_axis(Axis(0), b)
                .index_axis(Axis(0), c)
                .to_owned()
                .into_shape((input_matrix.shape()[2], input_matrix.shape()[3]))
                .unwrap();

            // Apply padding to the height-width input matrix
            let padded_matrix = pad_matrix_2d(&h_w_matrix, &pads)?;

            // Extract the dimensions of the padded matrix
            let (padded_rows, padded_cols) = padded_matrix.dim();

            // Calculate output dimensions
            let output_rows = (padded_rows - kernel_height) / stride_height + 1;
            let output_cols = (padded_cols - kernel_width) / stride_width + 1;

            // Parallelize the pooling operation
            let pooled_matrix = (0..output_rows)
                .into_par_iter()
                .map(|i| {
                    let mut row = Vec::with_capacity(output_cols);

                    for j in 0..output_cols {
                        let start_row = i * stride_height;
                        let start_col = j * stride_width;
                        let end_row = start_row + kernel_height;
                        let end_col = start_col + kernel_width;

                        // Extract the corresponding patch from the padded matrix
                        let patch = padded_matrix.slice(s![start_row..end_row, start_col..end_col]);

                        // Calculate the max value within the patch
                        let max_value = patch.fold(std::f32::NEG_INFINITY, |acc, &x| acc.max(x));

                        row.push(max_value);
                    }

                    row
                })
                .collect::<Vec<_>>();

            pooled_results.push(pooled_matrix);
        }
    }

    // Convert the collected rows into an ArrayD
    let pooled_tensor = ArrayD::from_shape_vec(
        IxDyn(&[
            batch_size,
            channels,
            pooled_results[0].len(),
            pooled_results[0][0].len(),
        ]),
        pooled_results.into_iter().flatten().flatten().collect(),
    )
    .map_err(|_| OnnxError::ShapeError("Failed to create output tensor".to_string()));

    pooled_tensor
}
