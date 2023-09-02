use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use rand::{Rng, SeedableRng};

/// `dropout` - ONNX Node Implementation for Dropout
///
/// Executes the dropout operation, which randomly nullifies a portion of the input units during
/// training to mitigate overfitting. This operation might consider an optional ratio indicating
/// the proportion of input units to drop. During inference (with training mode turned off), the
/// function omits dropout and scales the outputs using the retained ratio.
///
/// # Arguments
///
/// * `input` - A reference to the tensor to undergo dropout.
/// * `initializers` - Optional initializers:
///     * The first parameter (if provided) outlines the dropout ratio (defaults to 0.5 if not given).
///     * The second parameter (if given) indicates the model's mode (1 for training, 0 for inference).
/// * `node` - A reference to the ONNX NodeProto with node-specific data and attributes, such as the random seed.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Returns the tensor after applying dropout, or
///   an error (`OnnxError`) if any stage of the operation encounters an issue.
///
/// # Errors
///
/// Potential errors include:
/// * Conversion from `TensorProto` to ndarray not succeeding.
/// * Issues or invalid values during attribute extraction.
///
/// # Example
///
/// ```rust
/// let dropout_output = dropout(&input_tensor, Some(&params_tensors), &node);
/// ```
pub fn dropout(
    input: &TensorProto,
    initializers: Option<&Vec<&TensorProto>>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Convert TensorProto to ndarray
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    let (ratio, training_mode) = match initializers {
        Some(tensor_protos) => {
            let ratio = tensor_protos
                .get(0)
                .and_then(|tp| tp.get_float_data().get(0))
                .cloned()
                .unwrap_or(0.5);
            let training_mode = tensor_protos
                .get(1)
                .and_then(|tp| tp.get_int64_data().get(0))
                .cloned()
                .unwrap_or(0);

            (ratio, training_mode)
        }
        None => (0.5, 0),
    };

    if training_mode == 0 {
        let result = convert_to_output_tensor(node, input_nd_array.clone());
        return result;
    }

    let attributes = extract_attributes(node.get_attribute())?;
    let seed = get_int_attribute(&attributes, "seed", Some(rand::thread_rng().gen()))?;

    // Compute the scale
    let scale = 1. / (1. - ratio);

    let shape = input_nd_array.shape();
    let feature_shape: Vec<_> = shape[1..].to_vec();

    // Initialize the RNG with the provided seed
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);

    // Generate the random mask for a single batch element
    let mask_len = feature_shape.iter().product::<usize>();
    let single_mask: ArrayD<bool> =
        Array::from_iter(std::iter::repeat_with(|| rng.gen::<f32>() >= ratio).take(mask_len))
            .into_shape(feature_shape)
            .unwrap();

    // Convert single mask to the same shape as input, but repeating it for each batch element
    let mask = single_mask.broadcast(shape.to_vec()).unwrap();

    // Element-wise multiply the scaled input tensor by the mask
    let result = input_nd_array.mapv(|x| x * scale) * mask.mapv(|x| if x { 1.0 } else { 0.0 });

    // Convert the results back to TensorProto
    convert_to_output_tensor(node, result)
}
