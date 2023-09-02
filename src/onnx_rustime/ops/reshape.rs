use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, tensor_proto_to_ndarray,
};
use ndarray::prelude::*;

fn reshape_single_tensor(
    input: ArrayViewD<f32>,
    shape: &Vec<isize>,
    allow_zero: i64,
) -> Result<Array<f32, IxDyn>, OnnxError> {
    let mut inferred_dim = None;
    let mut target_shape = shape.clone();

    for (i, dim) in target_shape.iter_mut().enumerate() {
        if *dim == -1 {
            if inferred_dim.is_some() {
                return Err(OnnxError::ShapeMismatch(
                    "More than one inferred dimension!".into(),
                ));
            }
            inferred_dim = Some(i);
        } else if *dim == 0 {
            if allow_zero == 0 {
                *dim = input.shape()[i] as isize;
            }
        }
    }

    if let Some(idx) = inferred_dim {
        let product_of_dims: isize = target_shape.iter().filter(|&&dim| dim != -1).product();
        println!("idx: {:?}", idx);
        target_shape[idx] = (input.len() as isize) / product_of_dims;
    }

    Ok(input
        .into_shape(target_shape.iter().map(|&x| x as usize).collect::<Vec<_>>())
        .unwrap()
        .to_owned())
}

fn reshape_with_batches(
    input: &TensorProto,
    parameter: &TensorProto,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Extract node attributes.
    let attributes = extract_attributes(node.get_attribute())?;
    let allow_zero: i64 = get_int_attribute(&attributes, "allow_zero", Some(0))?;

    // Retrieve the data tensor either from `inputs` or from `initializers`.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(&input)?;

    // Determine the shape tensor.
    let shape_tensor = parameter;

    let mut target_shape: Vec<isize> = tensor_proto_to_ndarray::<i64>(shape_tensor)?
        .into_raw_vec()
        .iter()
        .map(|&x| x as isize)
        .collect();

    target_shape[0] *= input_nd_array.shape()[0] as isize;

    // Reshape each batch
    let reshaped_batches =
        reshape_single_tensor(input_nd_array.view(), &target_shape, allow_zero).unwrap();

    convert_to_output_tensor(node, reshaped_batches)
}

fn reshape_without_batches(
    initializers: &Vec<&TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Extract node attributes.
    let attributes = extract_attributes(node.get_attribute())?;
    let allow_zero: i64 = get_int_attribute(&attributes, "allow_zero", Some(0))?;

    let input_nd_array = tensor_proto_to_ndarray::<f32>(initializers[0])?;
    let target_shape: Vec<isize> = tensor_proto_to_ndarray::<i64>(initializers[1])?
        .into_raw_vec()
        .iter()
        .map(|&x| x as isize)
        .collect();

    let reshaped = reshape_single_tensor(input_nd_array.view(), &target_shape, allow_zero)?;
    convert_to_output_tensor(node, reshaped)
}

/// `reshape` - ONNX Node Implementation for Tensor Reshaping
///
/// The `reshape` operation provides functionality akin to `numpy.reshape`, allowing for the alteration
/// of tensor dimensions while preserving its data.
///
/// # Arguments
///
/// * `inputs` - An optional reference to the input tensor. If provided, this tensor will be reshaped.
/// * `initializers` - Contains tensors essential for the reshaping process. If `inputs` is `None`,
///   the first tensor in `initializers` will be considered the data tensor. The subsequent tensor in
///   `initializers` delineates the desired output shape.
/// * `node` - A reference to the ONNX NodeProto containing node-specific attributes and directives
///   for the reshaping operation.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the reshaped tensor. In the event of an issue
///   during the reshaping process, an error (`OnnxError`) will be returned.
///
/// # Errors
///
/// Potential issues that can arise:
/// * The reshaped tensor's dimensions do not match the defined shape.
/// * Encountering incompatible attributes during the reshaping.
///
/// # Notes
///
/// The core principles of the reshaping process include:
/// 1. Only one dimension in the new shape can have a value of -1. In this scenario, the dimension's
///    value is deduced from the tensor's size and any remaining dimensions.
/// 2. A dimension might be assigned a value of 0. If the `allowzero` attribute is unset, the dimension's
///    original value remains unchanged (sourced from the input tensor). If `allowzero` is active, and
///    the new shape contains a 0, this dimension will be explicitly set to zero.
/// 3. The total number of elements in both the input tensor's shape and the output tensor's shape must be identical.
///
/// Note that specifying a shape that includes both a 0 and a -1 value is invalid when the `allowzero`
/// attribute is activated.
pub fn reshape(
    inputs: Option<&TensorProto>,
    initializers: &Vec<&TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    if initializers.len() == 2 {
        reshape_without_batches(initializers, node)
    } else {
        reshape_with_batches(inputs.unwrap(), initializers[0], node)
    }
}
