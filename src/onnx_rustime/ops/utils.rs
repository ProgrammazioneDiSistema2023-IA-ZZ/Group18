#![allow(dead_code)]

/*
ONNX Operations Utility File

This utility file provides a set of functionalities tailored for working with the Open Neural Network Exchange (ONNX) format.
The utilities encompass a broad range of operations, including tensor data extraction, conversion, padding, stacking, and attribute handling.

Key Features:
- TensorType Trait: Defines a common interface for various tensor data types. It facilitates both data extraction from tensors and conversion of arrays into tensor data.
- Data Extraction: Comprehensive implementations are provided for extracting tensor data for various primitive types (f32, i32, i64, String). These methods handle both direct and raw data formats.
- Tensor Conversion: Utility functions are provided for converting between NDArrays and TensorProtos. These are essential for interfacing between ONNX and computational backends.
- Attribute Handling: A set of utilities to extract and categorize attributes from ONNX nodes. This provides a structured way to access attributes by their names and types.
- Matrix Padding: Functions to pad 2D and 3D matrices, a common operation in neural network layers.
- Batch Stacking: Allows stacking of tensors along a new batch dimension, useful for batch processing of data.

Errors:
The utility functions often return results wrapped in the Result type. In case of errors, a custom OnnxError type provides detailed information about the cause of the failure, helping in diagnostics and troubleshooting.

Usage:
To use these utilities, make sure to import the necessary dependencies. The functions and traits are designed to be generic and reusable across various ONNX models and backends.
It's recommended to refer to function-specific documentation for detailed information on parameters, return types, and examples.

Contribution:
This utility file is open for enhancements. If there are additional ONNX operations or features you'd like to see, consider contributing or raising a feature request.
*/

use crate::onnx_rustime::backend::helper::{make_tensor, Attribute, OnnxError, TensorValue};
use crate::onnx_rustime::backend::parser::{parse_raw_data_as_floats, parse_raw_data_as_ints64};
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{
    AttributeProto, AttributeProto_AttributeType, GraphProto, NodeProto, TensorProto,
};
use ndarray::*;
use std::collections::HashMap;

/// `TensorType` defines a trait for data types used in Tensors.
///
/// This trait provides methods for extracting data from a tensor and
/// converting an array to tensor data.
pub trait TensorType {
    /// Represents the specific type of data the tensor holds.
    type DataType;

    /// Extracts data from a given tensor and checks it against an expected length.
    ///
    /// # Arguments
    ///
    /// * `tensor` - A reference to the tensor from which data needs to be extracted.
    /// * `expected_len` - The expected length of the data.
    ///
    /// # Returns
    ///
    /// * `Result<ArrayD<Self::DataType>, OnnxError>` - An array of data if successful, or an error.
    fn extract_data(
        tensor: &TensorProto,
        expected_len: usize,
    ) -> Result<ArrayD<Self::DataType>, OnnxError>;

    /// Converts a given array to tensor data.
    ///
    /// # Arguments
    ///
    /// * `array` - The array to be converted.
    ///
    /// # Returns
    ///
    /// * `TensorValue` - The converted tensor value.
    fn to_tensor_data(array: ArrayD<Self::DataType>) -> TensorValue;
}

/// Implementation of `TensorType` for `f32` data type.
impl TensorType for f32 {
    type DataType = f32;

    fn extract_data(
        tensor: &TensorProto,
        expected_len: usize,
    ) -> Result<ArrayD<Self::DataType>, OnnxError> {
        // Extract shape from the tensor.
        let shape: Vec<usize> = tensor.get_dims().iter().map(|&dim| dim as usize).collect();

        // Check if float_data is present and matches the expected length.
        if !tensor.float_data.is_empty() && tensor.float_data.len() == expected_len {
            ArrayD::from_shape_vec(shape, tensor.float_data.clone())
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else if !tensor.raw_data.is_empty() {
            // Parse raw data as floats.
            let data = parse_raw_data_as_floats(&tensor.raw_data);
            ArrayD::from_shape_vec(shape, data).map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for FLOAT type".to_string(),
            ))
        }
    }

    fn to_tensor_data(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::Float(array.into_dyn().into_raw_vec())
    }
}

/// Implementation of `TensorType` for `i32` data type.
impl TensorType for i32 {
    type DataType = i32;

    fn extract_data(
        tensor: &TensorProto,
        expected_len: usize,
    ) -> Result<ArrayD<Self::DataType>, OnnxError> {
        let shape: Vec<usize> = tensor.get_dims().iter().map(|&dim| dim as usize).collect();

        if !tensor.int32_data.is_empty() && tensor.int32_data.len() == expected_len {
            ArrayD::from_shape_vec(shape, tensor.int32_data.clone())
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for INT32 type".to_string(),
            ))
        }
    }

    fn to_tensor_data(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::Int32(array.into_dyn().into_raw_vec())
    }
}

/// Implementation of `TensorType` for `i64` data type.
impl TensorType for i64 {
    type DataType = i64;

    fn extract_data(
        tensor: &TensorProto,
        expected_len: usize,
    ) -> Result<ArrayD<Self::DataType>, OnnxError> {
        // Extract shape from the tensor.
        let shape: Vec<usize> = tensor.get_dims().iter().map(|&dim| dim as usize).collect();

        if !tensor.int64_data.is_empty() && tensor.int64_data.len() == expected_len {
            ArrayD::from_shape_vec(shape, tensor.int64_data.clone())
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else if !tensor.raw_data.is_empty() {
            // Parse raw data as floats.
            let data = parse_raw_data_as_ints64(&tensor.raw_data);
            ArrayD::from_shape_vec(shape, data).map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for INT64 type".to_string(),
            ))
        }
    }

    fn to_tensor_data(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::Int64(array.into_dyn().into_raw_vec())
    }
}

/// Implementation of `TensorType` for `String` data type.
impl TensorType for String {
    type DataType = String;

    fn extract_data(
        tensor: &TensorProto,
        expected_len: usize,
    ) -> Result<ArrayD<Self::DataType>, OnnxError> {
        // Extract shape from the tensor.
        let shape: Vec<usize> = tensor.get_dims().iter().map(|&dim| dim as usize).collect();

        // Check if string_data is present and matches the expected length.
        if !tensor.string_data.is_empty() && tensor.string_data.len() == expected_len {
            let string_data = tensor
                .string_data
                .iter()
                .map(|s| String::from_utf8_lossy(s).to_string())
                .collect::<Vec<_>>();
            ArrayD::from_shape_vec(shape, string_data)
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for STRING type".to_string(),
            ))
        }
    }

    fn to_tensor_data(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::String(array.into_dyn().into_raw_vec())
    }
}

/// Converts an NDArray to a TensorProto.
///
/// # Arguments
///
/// * `result` - The NDArray to be converted.
/// * `output_name` - The desired name for the resulting TensorProto.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - The converted TensorProto or an error.
pub fn ndarray_to_tensor_proto<T: TensorType>(
    result: ArrayD<T::DataType>,
    output_name: &str,
) -> Result<TensorProto, OnnxError> {
    // Extract dimensions from the NDArray and convert them to i64.
    let tensor_dims = result
        .shape()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();

    // Convert NDArray data to tensor data.
    let tensor_data = T::to_tensor_data(result);

    // Construct the TensorProto.
    Ok(make_tensor(Some(output_name), tensor_dims, tensor_data))
}

/// Converts the result into a `TensorProto` using the output name from the given node.
///
/// # Arguments
/// * `node`: The `NodeProto` that contains information about the output name.
/// * `result`: The resultant array to be converted into a `TensorProto`.
///
/// # Returns
/// * `TensorProto`: The resultant tensor.
/// * `OnnxError`: An error indicating if the output name is missing or there's an error during conversion.
pub fn convert_to_output_tensor(
    node: &NodeProto,
    result: ArrayD<f32>,
) -> Result<TensorProto, OnnxError> {
    let output_name = node
        .get_output()
        .get(0)
        .ok_or(OnnxError::InternalError("Output name missing".to_string()))?;

    ndarray_to_tensor_proto::<f32>(result, output_name)
}

// Constants representing different data types in TensorProto.
// They are mapped to the TensorProto data field.
const DATA_TYPE_FLOAT: i32 = 1;
const DATA_TYPE_INT32: i32 = 5;
const DATA_TYPE_STRING: i32 = 6;
const DATA_TYPE_INT64: i32 = 7;
const DATA_TYPE_RAW: i32 = 9;

/// Converts a TensorProto to an NDArray.
///
/// # Arguments
///
/// * `tensor` - The TensorProto to be converted.
///
/// # Returns
///
/// * `Result<ArrayD<T::DataType>, OnnxError>` - The converted NDArray or an error.
pub fn tensor_proto_to_ndarray<T: TensorType>(
    tensor: &TensorProto,
) -> Result<ArrayD<T::DataType>, OnnxError> {
    // Calculate the expected length based on the dimensions of the tensor.
    let expected_len: usize = tensor.get_dims().iter().map(|&dim| dim as usize).product();

    // Match on the data type of the tensor and extract the data accordingly.
    match Some(tensor.get_data_type()) {
        Some(DATA_TYPE_FLOAT) => T::extract_data(tensor, expected_len),
        Some(DATA_TYPE_INT32) => T::extract_data(tensor, expected_len),
        Some(DATA_TYPE_STRING) => T::extract_data(tensor, expected_len),
        Some(DATA_TYPE_INT64) => T::extract_data(tensor, expected_len),
        Some(DATA_TYPE_RAW) => T::extract_data(tensor, expected_len),
        _ => Err(OnnxError::UnsupportedOperation(format!(
            "Unsupported data type: {}",
            tensor.get_data_type()
        ))),
    }
}

/// Extracts raw data from a TensorProto.
///
/// # Arguments
///
/// * `tensor` - The TensorProto containing the raw data.
/// * `expected_len` - The expected length of the extracted data.
///
/// # Returns
///
/// * `Result<Vec<f32>, OnnxError>` - The extracted raw data or an error.
fn extract_raw_data(tensor: &TensorProto, expected_len: usize) -> Result<Vec<f32>, OnnxError> {
    if !tensor.raw_data.is_empty() {
        let data = parse_raw_data_as_floats(&tensor.raw_data);
        if data.len() == expected_len {
            Ok(data)
        } else {
            Err(OnnxError::ShapeMismatch(format!(
                "Data length mismatch in RAW data: expected {} but got {}",
                expected_len,
                data.len()
            )))
        }
    } else {
        Err(OnnxError::MissingInput(
            "No data found for RAW type".to_string(),
        ))
    }
}

/// Extracts attributes from a list of AttributeProtos and maps them to a HashMap.
///
/// This function processes the attributes and categorizes them based on their type,
/// e.g., Float, Int, String, etc. The resulting map provides a structured way to access
/// these attributes by their names.
///
/// # Arguments
///
/// * `attributes` - A slice of AttributeProtos to extract attributes from.
///
/// # Returns
///
/// * `Result<HashMap<String, Attribute<String>>, OnnxError>` - A HashMap of extracted attributes or an error.
pub fn extract_attributes(
    attributes: &[AttributeProto],
) -> Result<HashMap<String, Attribute<String>>, OnnxError> {
    let mut attribute_map = HashMap::new();

    for attr in attributes {
        // Extract the attribute name.
        let key = attr.get_name().to_string();

        // Match on the attribute type and extract the value accordingly.
        let value = match attr.get_field_type() {
            AttributeProto_AttributeType::FLOAT => Attribute::Float(attr.get_f()),
            AttributeProto_AttributeType::INT => Attribute::Int(attr.get_i()),
            AttributeProto_AttributeType::STRING => {
                Attribute::String(String::from_utf8(attr.get_s().to_vec()).map_err(|_| {
                    OnnxError::ConversionError(
                        "Failed to convert bytes to UTF-8 string".to_string(),
                    )
                })?)
            }
            AttributeProto_AttributeType::TENSOR => Attribute::Tensor(attr.get_t().clone()),
            AttributeProto_AttributeType::GRAPH => Attribute::Graph(attr.get_g().clone()),
            AttributeProto_AttributeType::FLOATS => Attribute::Floats(attr.get_floats().to_vec()),
            AttributeProto_AttributeType::INTS => Attribute::Ints(attr.get_ints().to_vec()),
            AttributeProto_AttributeType::STRINGS => Attribute::Strings(
                attr.get_strings()
                    .iter()
                    .map(|s| {
                        String::from_utf8(s.to_vec()).map_err(|_| {
                            OnnxError::ConversionError(
                                "Failed to convert bytes to UTF-8 string".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            AttributeProto_AttributeType::TENSORS => {
                Attribute::Tensors(attr.get_tensors().to_vec())
            }
            AttributeProto_AttributeType::GRAPHS => Attribute::Graphs(attr.get_graphs().to_vec()),
            _ => {
                return Err(OnnxError::UnsupportedOperation(
                    "Unsupported attribute type".to_string(),
                ))
            }
        };

        // Insert the extracted attribute into the map.
        attribute_map.insert(key, value);
    }

    Ok(attribute_map)
}

// The following functions provide a convenient way to extract specific types of attributes
// from the HashMap generated by `extract_attributes`. If the attribute does not exist or
// is of the wrong type, an error is returned. Optionally, a default value can be provided.

/// Retrieves a Float (`f32`) attribute.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default value to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<f32, OnnxError>` - The extracted Float attribute or an error.
pub fn get_float_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<f32>,
) -> Result<f32, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_float())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a list of Float (`f32`) attributes.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default list of Ints to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<Vec<f32>, OnnxError>` - The extracted list of Int attributes or an error.
pub fn get_floats_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<Vec<f32>>,
) -> Result<Vec<f32>, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_floats().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a Int (`i64`) attribute.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default value to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<i64, OnnxError>` - The extracted Float attribute or an error.
pub fn get_int_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<i64>,
) -> Result<i64, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_int())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a list of Int (`i64`) attributes.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default list of Ints to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<Vec<i64>, OnnxError>` - The extracted list of Int attributes or an error.
pub fn get_ints_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<Vec<i64>>,
) -> Result<Vec<i64>, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_ints().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a String attribute.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default String to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<String, OnnxError>` - The extracted String attribute or an error.
pub fn get_string_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<String>,
) -> Result<String, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_string().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a list of String attributes.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default list of Strings to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<Vec<String>, OnnxError>` - The extracted list of String attributes or an error.
pub fn get_strings_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<Vec<String>>,
) -> Result<Vec<String>, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_strings().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a TensorProto attribute.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default TensorProto to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - The extracted TensorProto attribute or an error.
pub fn get_tensor_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<TensorProto>,
) -> Result<TensorProto, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_tensor().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a list of TensorProto attributes.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default list of TensorProtos to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<Vec<TensorProto>, OnnxError>` - The extracted list of TensorProto attributes or an error.
pub fn get_tensors_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<Vec<TensorProto>>,
) -> Result<Vec<TensorProto>, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_tensors().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a GraphProto attribute.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default GraphProto to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<GraphProto, OnnxError>` - The extracted GraphProto attribute or an error.
pub fn get_graph_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<GraphProto>,
) -> Result<GraphProto, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_graph().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a list of GraphProto attributes.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default list of GraphProtos to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<Vec<GraphProto>, OnnxError>` - The extracted list of GraphProto attributes or an error.
pub fn get_graphs_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<Vec<GraphProto>>,
) -> Result<Vec<GraphProto>, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_graphs().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Pad a 2D matrix with specified values.
///
/// This function takes a 2D matrix and pads it according to the specified padding values.
///
/// # Arguments
///
/// * `matrix` - A reference to the input 2D matrix.
/// * `pads` - A reference to a vector containing padding values in the order [top, bottom, left, right].
///
/// # Returns
///
/// Returns a Result containing the padded 2D matrix if successful, or an OnnxError if there's an issue.
///
/// # Examples
///
/// ```rust
/// use ndarray::array;
/// let matrix = array![[1.0, 2.0], [3.0, 4.0]];
/// let pads = vec![1, 1, 1, 1];
/// let padded_matrix = pad_matrix_2d(&matrix, &pads).unwrap();
/// ```
pub fn pad_matrix_2d(matrix: &Array2<f32>, pads: &Vec<i64>) -> Result<Array2<f32>, OnnxError> {
    // Extract padding values
    let top = pads[0] as usize;
    let left = pads[1] as usize;
    let bottom = pads[2] as usize;
    let right = pads[3] as usize;

    // If no padding is needed, return a clone of the input matrix
    if top == 0 && bottom == 0 && left == 0 && right == 0 {
        return Ok(matrix.clone());
    }

    // Calculate the shape of the padded matrix
    let padded_shape = (
        matrix.shape()[0] + top + bottom,
        matrix.shape()[1] + left + right,
    );

    // Create a new matrix filled with zeros for padding
    let mut padded_matrix = Array2::zeros(padded_shape);

    // Slice the padded matrix and assign the input matrix values to it
    padded_matrix
        .slice_mut(s![
            top..padded_shape.0 - bottom,
            left..padded_shape.1 - right
        ])
        .assign(&matrix);

    // Return the padded matrix
    Ok(padded_matrix)
}

pub fn pad_matrix_3d(matrix: &Array3<f32>, pads: &Vec<i64>) -> Result<Array3<f32>, OnnxError> {
    let c = matrix.shape()[0];

    // Vector to store the actual padded slices (owned data)
    let mut padded_data = Vec::with_capacity(c);

    for depth in 0..c {
        // Extract the 2D slice from the 3D array
        let slice = matrix.index_axis(ndarray::Axis(0), depth).to_owned();

        // Pad the slice using the existing function
        let padded = pad_matrix_2d(&slice, pads)?;

        // Push the padded slice into the vector
        padded_data.push(padded);
    }

    // Derive a vector of views from our owned data
    let padded_slices: Vec<_> = padded_data.iter().map(|array| array.view()).collect();

    // Now, directly use the stack function with the slice of views
    ndarray::stack(ndarray::Axis(0), &padded_slices[..])
        .map_err(|_| OnnxError::InternalError("Failed to stack matrices.".to_string()))
}

/// Stacks the provided tensors along the batch dimension.
///
/// This function takes a vector of tensors and stacks them along a new batch dimension.
/// The resulting tensor will have one additional dimension (compared to the input tensors),
/// where the size of the new dimension equals the number of input tensors.
///
/// # Arguments
///
/// * `tensors` - A vector of tensors to be stacked along the batch dimension.
///
/// # Returns
///
/// Returns a `Result` containing:
/// * An `ArrayD` tensor that combines all input tensors along the batch dimension.
/// * An `OnnxError` if stacking fails, or if reshaping the output tensor fails.
///
/// # Type Parameters
///
/// * `T` - The data type of the tensor elements.
/// * `D` - The dimension type of the input tensors.
///
/// # Constraints
///
/// * `T` must implement the `Clone` trait to enable duplication of tensor elements.
/// * `D` must implement the `Dimension` trait to represent the shape of the tensor.
///
/// # Errors
///
/// Returns an error of type `OnnxError::ShapeError` if:
/// * The tensors cannot be stacked along the batch dimension.
/// * Reshaping the output tensor fails after stacking.
///
/// # Examples
///
/// ```rust
/// # use your_imports_here;
/// let tensor1 = array![[1.0, 2.0], [3.0, 4.0]];
/// let tensor2 = array![[5.0, 6.0], [7.0, 8.0]];
///
/// let stacked = stack_along_batch_dimension(vec![tensor1, tensor2]).unwrap();
/// assert_eq!(stacked.dim(), (2, 2, 2));
/// ```
pub fn stack_along_batch_dimension<T, D>(
    tensors: Vec<ArrayBase<OwnedRepr<T>, D>>,
) -> Result<ArrayD<T>, OnnxError>
where
    T: Clone,
    D: Dimension,
{
    // Create views of the tensors
    let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();

    // Stack along the batch dimension
    let stacked_output = ndarray::stack(Axis(0), &views).map_err(|_| {
        OnnxError::ShapeError("Failed to stack tensors along batch dimension.".to_string())
    })?;

    // Construct the result dimensions dynamically
    let mut result_dims: Vec<usize> = Vec::with_capacity(stacked_output.ndim());
    result_dims.push(tensors.len()); // First dimension is the batch size
    result_dims.extend_from_slice(stacked_output.shape().split_at(1).1); // Skip the first dimension as we already added it

    let reshaped_output =
        ArrayD::from_shape_vec(IxDyn(&result_dims), stacked_output.into_iter().collect())
            .map_err(|_| OnnxError::ShapeError("Failed to reshape stacked output.".to_string()))?;

    Ok(reshaped_output)
}
