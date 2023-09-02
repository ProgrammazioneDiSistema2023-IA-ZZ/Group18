use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{NodeProto, TensorProto};
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_float_attribute, get_int_attribute,
    tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use rayon::prelude::*;

pub enum OperationMode {
    Gemm,
    Matmul,
}

/// `gemm` - ONNX Node Implementation for General Matrix Multiplication (GEMM)
///
/// Performs General Matrix Multiplication (GEMM) as defined by the equation:
/// \( Y = \alpha \times A' \times B' + \beta \times C \)
///
/// Here:
/// - \( A' \) is the transpose of \( A \) if `transA` is true, otherwise it's \( A \).
/// - \( B' \) is the transpose of \( B \) if `transB` is true, otherwise it's \( B \).
/// - `alpha` and `beta` are scalar multipliers.
/// - For plain matrix multiplication (if node type is "MatMul"):
///     * `alpha` defaults to 1 (no scaling).
///     * `beta` defaults to 0 (effectively disregarding \( C \)).
///
/// This operation supports unidirectional broadcasting where tensor \( C \) should be
/// unidirectionally broadcastable to the shape of \( A \times B \).
///
/// # Arguments
///
/// * `inputs`: A vector containing references to the input tensors `A`, `B`,
///   and optionally `C` if the operation type is GEMM.
/// * `initializers`: An optional vector of initializers, currently unused.
/// * `node`: A reference to the ONNX `NodeProto` that might have node-specific attributes
///   like `alpha`, `beta`, `transA`, and `transB`. The node type determines whether the operation
///   is GEMM or plain matrix multiplication (MatMul).
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>`: Outputs the resultant tensor from the matrix operation,
///   or returns an error (`OnnxError`) in case of issues such as missing input tensors or
///   shape mismatches.
///
/// # Example
///
/// ```rust
/// let result_tensor = gemm(&input_tensors, None, &node);
/// ```
pub fn gemm(
    inputs: &Vec<&TensorProto>,
    initializers: Option<&Vec<&TensorProto>>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Extract operation attributes and decide the operation mode.
    let attributes = extract_attributes(node.get_attribute())?;
    let mode = determine_mode(&node.op_type)?;

    // Fetch operation attributes.
    let alfa: f32 = get_float_attribute(&attributes, "alpha", Some(1.0))?;
    let beta: f32 = get_float_attribute(&attributes, "beta", Some(0.0))?;
    let trans_a: i64 = get_int_attribute(&attributes, "transA", Some(0))?;
    let trans_b: i64 = get_int_attribute(&attributes, "transB", Some(0))?;

    // Merge inputs and initializers.
    let mut merged_tensors: Vec<&TensorProto> = inputs.clone();
    if let Some(params) = initializers {
        merged_tensors.extend(params);
    }

    // Convert TensorProtos to ndarray using the merged list.
    let mut a = tensor_proto_to_ndarray::<f32>(get_tensor(&merged_tensors, 0, "A")?)?;
    let mut b = tensor_proto_to_ndarray::<f32>(get_tensor(&merged_tensors, 1, "B")?)?;

    // Transpose matrices based on attributes
    if trans_a == 1 {
        a = a.t().to_owned();
    }
    if trans_b == 1 {
        b = b.t().to_owned();
    }

    // Perform the matrix multiplication.
    let mut result = matrix_multiply(&a, &b).ok_or(OnnxError::InternalError(
        "Failed to multiply matrices".to_string(),
    ))?;
    result.mapv_inplace(|x| x * alfa);

    // Handle the case for GEMM operation where there's an optional C matrix.
    if let OperationMode::Gemm = mode {
        if let Some(c_tensor_proto) = inputs.get(2) {
            let mut c_array = tensor_proto_to_ndarray::<f32>(c_tensor_proto)?;
            c_array.mapv_inplace(|x| x * beta);

            // Ensure both matrices are broadcast-compatible.
            if result.shape() != c_array.shape() {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Expected shape {:?}, but got {:?}",
                    result.shape(),
                    c_array.shape()
                )));
            }
            result += &c_array;
        }
    }

    convert_to_output_tensor(node, result)
}

/// Determines the operation mode based on the operation type.
fn determine_mode(op_type: &str) -> Result<OperationMode, OnnxError> {
    match op_type {
        "Gemm" => Ok(OperationMode::Gemm),
        "MatMul" => Ok(OperationMode::Matmul),
        _ => Err(OnnxError::InternalError(format!(
            "Unsupported operation: {}",
            op_type
        ))),
    }
}

/// Retrieves a tensor from the input list by index and provides an error if missing.
fn get_tensor<'a>(
    inputs: &'a Vec<&'a TensorProto>,
    index: usize,
    name: &str,
) -> Result<&'a TensorProto, OnnxError> {
    inputs
        .get(index)
        .copied()
        .ok_or(OnnxError::MissingInput(name.to_string()))
}

fn matrix_multiply(a: &ArrayD<f32>, b: &ArrayD<f32>) -> Option<ArrayD<f32>> {
    let b_matrix = b
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .to_owned();

    match a.shape()[0] {
        1 => matrix_multiply_single(a, &b_matrix),
        _ => matrix_multiply_batched(a, &b_matrix),
    }
}

/// Handle the batched matrix multiplication.
fn matrix_multiply_batched(
    a: &ArrayD<f32>,
    b_matrix: &ndarray::Array2<f32>,
) -> Option<ArrayD<f32>> {
    let shape = a.shape();
    let batch_size = shape[0];

    // Parallel processing of the batch
    let result_list: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let a_slice = a.slice(s![i, ..]);
            a_slice.dot(b_matrix)
        })
        .collect();

    let views: Vec<_> = result_list.iter().map(|arr| arr.view()).collect();
    let result = ndarray::stack(Axis(0), &views[..]).unwrap();

    Some(result.into_dyn())
}

/// Handle a single matrix multiplication.
fn matrix_multiply_single(a: &ArrayD<f32>, b_matrix: &ndarray::Array2<f32>) -> Option<ArrayD<f32>> {
    let a_2d = a.to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let result = a_2d.dot(b_matrix);
    Some(result.into_dyn())
}
