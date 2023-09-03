#![allow(dead_code)]
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::{ndarray_to_tensor_proto, tensor_proto_to_ndarray};
use protobuf::{ProtobufEnum, RepeatedField};
use std::collections::HashMap;
use crate::onnx_rustime::shared::{MODEL_NAME, Model};

/// Represents the various types of errors that can occur within the ONNX runtime.
///
/// This enum provides detailed error variants that capture specific failure points within the
/// ONNX processing pipeline. Each variant provides a description, often encapsulating
/// additional information about the root cause of the error.
#[derive(Debug)]
pub enum OnnxError {
    /// Indicates that a required attribute was not found.
    ///
    /// The contained `String` provides the name or identifier of the missing attribute.
    AttributeNotFound(String),

    /// Represents generic internal errors that might occur during processing.
    ///
    /// The contained `String` provides a description or message detailing the nature of the internal error.
    InternalError(String),

    /// Indicates an error that occurred during data type conversion.
    ///
    /// The contained `String` provides additional information about the conversion that failed.
    ConversionError(String),

    /// Represents an error where an operation or functionality is not supported.
    ///
    /// The contained `String` provides details about the unsupported operation.
    UnsupportedOperation(String),

    /// Indicates a mismatch between expected and actual tensor shapes.
    ///
    /// The contained `String` provides details about the shape mismatch, such as the expected vs. actual dimensions.
    ShapeMismatch(String),

    /// Represents an error where an expected input tensor or data is missing.
    ///
    /// The contained `String` provides details about the missing input.
    MissingInput(String),

    /// Indicates an error due to invalid data or values.
    ///
    /// The contained `String` provides details about the nature of the invalid data.
    InvalidValue(String),

    /// Indicates an error related to tensor shape computations.
    ///
    /// The contained `String` provides details about the shape computation error.
    ShapeError(String),
}

macro_rules! set_optional {
    ($proto: ident . $setter: ident ( $val: ident ) ) => {
        if let Some($val) = $val {
            $proto.$setter($val.into())
        }
    };
}

macro_rules! set_repeated {
    ($proto: ident . $setter: ident ( $val: ident ) ) => {
        $proto.$setter(RepeatedField::from_vec($val))
    };
    ($proto: ident . $setter: ident ( $val: ident .into() ) ) => {
        $proto.$setter(RepeatedField::from_vec(
            $val.into_iter().map(Into::into).collect(),
        ))
    };
}

pub fn make_model<S: Into<String>, T: Into<i64>>(
    graph: GraphProto,
    opset_imports: Vec<OperatorSetIdProto>,
    domain: Option<S>,
    model_version: Option<T>,
    producer_name: Option<S>,
    producer_version: Option<S>,
    doc_string: Option<S>,
    metadata: Option<HashMap<String, String>>,
) -> ModelProto {
    let mut model_proto = ModelProto::new();

    model_proto.set_ir_version(Version::IR_VERSION.value() as i64);
    model_proto.set_graph(graph);
    model_proto.set_opset_import(RepeatedField::from_vec(if opset_imports.len() > 0 {
        opset_imports
    } else {
        let none_string: Option<String> = None;
        vec![make_opsetid(none_string, 3)]
    }));

    set_optional!(model_proto.set_domain(domain));
    set_optional!(model_proto.set_model_version(model_version));
    set_optional!(model_proto.set_producer_name(producer_name));
    set_optional!(model_proto.set_producer_version(producer_version));
    set_optional!(model_proto.set_doc_string(doc_string));

    if let Some(metadata) = metadata {
        model_proto.set_metadata_props(RepeatedField::from_vec(
            metadata
                .into_iter()
                .map(|(k, v)| {
                    let mut ss_proto = StringStringEntryProto::new();
                    ss_proto.set_key(k.into());
                    ss_proto.set_value(v.into());
                    ss_proto
                })
                .collect(),
        ));
    }
    model_proto
}

pub fn make_opsetid<S: Into<String>, T: Into<i64>>(
    domain: Option<S>,
    version: T,
) -> OperatorSetIdProto {
    let mut opsetid_proto = OperatorSetIdProto::new();

    set_optional!(opsetid_proto.set_domain(domain));
    opsetid_proto.set_version(version.into());
    opsetid_proto
}

pub fn make_graph<S: Into<String>>(
    nodes: Vec<NodeProto>,
    name: S,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    initializer: Vec<TensorProto>,
    doc_string: Option<S>,
) -> GraphProto {
    let mut graph_proto = GraphProto::new();

    graph_proto.set_name(name.into());

    set_repeated!(graph_proto.set_node(nodes));
    set_repeated!(graph_proto.set_input(inputs));
    set_repeated!(graph_proto.set_output(outputs));
    set_repeated!(graph_proto.set_initializer(initializer));
    set_optional!(graph_proto.set_doc_string(doc_string));

    graph_proto
}

pub fn make_node<S: Into<String>>(
    op_type: S,
    inputs: Vec<S>,
    outputs: Vec<S>,
    name: Option<S>,
    doc_string: Option<S>,
    domain: Option<S>,
    attributes: Vec<AttributeProto>,
) -> NodeProto {
    let mut node_proto = NodeProto::new();

    node_proto.set_op_type(op_type.into());

    set_repeated!(node_proto.set_input(inputs.into()));
    set_repeated!(node_proto.set_output(outputs.into()));
    set_optional!(node_proto.set_name(name));
    set_optional!(node_proto.set_domain(domain));
    set_optional!(node_proto.set_doc_string(doc_string));
    set_repeated!(node_proto.set_attribute(attributes));

    node_proto
}

#[derive(Debug)]
pub enum Attribute<S> {
    Float(f32),
    Floats(Vec<f32>),
    Int(i64),
    Ints(Vec<i64>),
    String(S),
    Strings(Vec<S>),
    Tensor(TensorProto),
    Tensors(Vec<TensorProto>),
    Graph(GraphProto),
    Graphs(Vec<GraphProto>),
}

impl<S: std::fmt::Display> Attribute<S> {
    pub fn as_float(&self) -> Option<f32> {
        if let Attribute::Float(value) = self {
            Some(*value)
        } else {
            None
        }
    }

    pub fn as_floats(&self) -> Option<&Vec<f32>> {
        if let Attribute::Floats(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        if let Attribute::Int(value) = self {
            Some(*value)
        } else {
            None
        }
    }

    pub fn as_ints(&self) -> Option<&Vec<i64>> {
        if let Attribute::Ints(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_string(&self) -> Option<&S> {
        if let Attribute::String(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_strings(&self) -> Option<&Vec<S>> {
        if let Attribute::Strings(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_tensor(&self) -> Option<&TensorProto> {
        if let Attribute::Tensor(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_tensors(&self) -> Option<&Vec<TensorProto>> {
        if let Attribute::Tensors(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_graph(&self) -> Option<&GraphProto> {
        if let Attribute::Graph(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_graphs(&self) -> Option<&Vec<GraphProto>> {
        if let Attribute::Graphs(ref value) = self {
            Some(value)
        } else {
            None
        }
    }
}

pub fn make_attribute<S: Into<String>, U: Into<Vec<u8>>>(
    name: S,
    attribute: Attribute<U>,
) -> AttributeProto {
    let mut attr_proto = AttributeProto::new();
    attr_proto.set_name(name.into());
    match attribute {
        Attribute::Float(val) => {
            attr_proto.set_f(val);
            attr_proto.set_field_type(AttributeProto_AttributeType::FLOAT);
        }
        Attribute::Floats(vals) => {
            attr_proto.set_floats(vals);
            attr_proto.set_field_type(AttributeProto_AttributeType::FLOATS);
        }
        Attribute::Int(val) => {
            attr_proto.set_i(val);
            attr_proto.set_field_type(AttributeProto_AttributeType::INT);
        }
        Attribute::Ints(vals) => {
            attr_proto.set_ints(vals);
            attr_proto.set_field_type(AttributeProto_AttributeType::INTS);
        }
        Attribute::String(val) => {
            attr_proto.set_s(val.into());
            attr_proto.set_field_type(AttributeProto_AttributeType::STRING);
        }
        Attribute::Strings(vals) => {
            attr_proto.set_strings(vals.into_iter().map(Into::into).collect());
            attr_proto.set_field_type(AttributeProto_AttributeType::STRINGS);
        }
        Attribute::Graph(val) => {
            attr_proto.set_g(val);
            attr_proto.set_field_type(AttributeProto_AttributeType::GRAPH);
        }
        Attribute::Graphs(vals) => {
            set_repeated!(attr_proto.set_graphs(vals));
            attr_proto.set_field_type(AttributeProto_AttributeType::GRAPHS);
        }
        Attribute::Tensor(val) => {
            attr_proto.set_t(val);
            attr_proto.set_field_type(AttributeProto_AttributeType::TENSOR);
        }
        Attribute::Tensors(vals) => {
            set_repeated!(attr_proto.set_tensors(vals));
            attr_proto.set_field_type(AttributeProto_AttributeType::TENSORS);
        }
    };
    attr_proto
}

#[allow(dead_code)]
pub enum TensorValue {
    Float(Vec<f32>),
    UInt8(Vec<u8>),
    Int8(Vec<i8>),
    UInt16(Vec<u16>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    String(Vec<String>),
    Bool(Vec<bool>),
    Double(Vec<f64>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
}

macro_rules! set_tensor_data {
  ($proto: ident, $vals: ident $(| $type: ident $proto_type: ident $setter: ident)+) => {
      match $vals {
          TensorValue::Bool(vals) => {
              $proto.set_int32_data(vals.into_iter().map(|v| if v { 1 } else { 0 }).collect());
              $proto.set_data_type(TensorProto_DataType::BOOL as i32);
          }
          $(TensorValue::$type(vals) => {
              $proto.$setter(vals.into_iter().map(Into::into).collect());
              $proto.set_data_type(TensorProto_DataType::$proto_type as i32);
          })+
      };
  }
}

pub fn make_tensor<S: Into<String>>(
    name: Option<S>,
    dims: Vec<i64>,
    vals: TensorValue,
) -> TensorProto {
    let mut tensor_proto = TensorProto::new();
    tensor_proto.set_dims(dims);
    set_optional!(tensor_proto.set_name(name));
    set_tensor_data!(tensor_proto, vals
        | Float   FLOAT   set_float_data
        | UInt8   UINT8   set_int32_data
        | Int8    INT8    set_int32_data
        | UInt16  UINT16  set_int32_data
        | Int16   INT16   set_int32_data
        | Int32   INT32   set_int32_data
        | String  STRING  set_string_data
        | UInt32  UINT32  set_uint64_data
        | UInt64   UINT64  set_uint64_data
        | Int64   INT64   set_int64_data
        | Double  DOUBLE  set_double_data
        // | Bool    BOOL    set_int32_data // no from -> special-cased
    );
    tensor_proto
}

pub enum Dimension {
    Value(i64),
    Param(String),
}

pub fn make_tensor_value_info<S: Into<String>>(
    name: S,
    elem_type: TensorProto_DataType,
    shape: Vec<Dimension>,
    doc_string: Option<S>,
) -> ValueInfoProto {
    let mut tensor_shape_proto = TensorShapeProto::new();

    tensor_shape_proto.set_dim(
        shape
            .into_iter()
            .map(|s| {
                let mut dim = TensorShapeProto_Dimension::new();
                match s {
                    Dimension::Value(v) => dim.set_dim_value(v),
                    Dimension::Param(p) => dim.set_dim_param(p),
                };
                dim
            })
            .collect(),
    );

    let mut tensor_type_proto = TypeProto_Tensor::new();
    tensor_type_proto.set_elem_type(elem_type as i32);
    tensor_type_proto.set_shape(tensor_shape_proto);

    let mut type_proto = TypeProto::new();
    type_proto.set_tensor_type(tensor_type_proto);

    let mut value_info_proto = ValueInfoProto::new();
    value_info_proto.set_name(name.into());
    value_info_proto.set_field_type(type_proto);
    set_optional!(value_info_proto.set_doc_string(doc_string));

    value_info_proto
}

pub fn duplicate_input_tensor(input: &TensorProto, n: usize) -> TensorProto {
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).unwrap();
    let input_shape: Vec<usize> = input
        .get_dims()
        .to_vec()
        .iter()
        .map(|&x| x as usize)
        .collect();

    let vec_input = input_nd_array.into_raw_vec();

    let mut duplicated_arrays = Vec::new();

    for _ in 0..n {
        duplicated_arrays.extend(vec_input.clone().iter());
    }

    let result_array = ndarray::ArrayD::<f32>::from_shape_vec(
        ndarray::IxDyn(&[n, input_shape[1], input_shape[2], input_shape[3]]),
        duplicated_arrays,
    )
    .unwrap();

    ndarray_to_tensor_proto::<f32>(result_array, input.get_name()).unwrap()
}

use ndarray::prelude::*;

fn logits_to_prob(logits: ArrayView1<f32>) -> Array1<f32> {
    let model_name = {
        let lock = MODEL_NAME.lock().unwrap();
        lock.clone()
    };

    match model_name {
        Model::ResNet => {
            let max = logits.fold(0. / 0., |m, &val| f32::max(m, val)); // NaN-safe max
            let exps = logits.mapv(|x| (x - max).exp());
            let sum = exps.sum();
            
            exps / sum
        },
        Model::Mnist => {
            let max = logits.fold(0. / 0., |m, &val| f32::max(m, val)); // NaN-safe max
            let exps = logits.mapv(|x| (x - max).exp());
            let sum = exps.sum();
            
            exps / sum
        },
        _ => logits.to_owned()
    }
}

pub fn find_top_5_peak_classes(output: &ArrayD<f32>) -> Option<Vec<Vec<(usize, f32)>>> {
    // Calculate total number of elements and the batch size
    let total_elements = output.len();
    let first_dim = output.shape()[0];
    let inferred_dim = total_elements / first_dim;

    // Reshape the tensor
    let reshaped = output.view().into_shape((first_dim, inferred_dim)).ok()?;

    // Apply softmax and find the top 5 peak classes for each batch
    let mut top_5_peak_classes = Vec::with_capacity(first_dim);

    for batch in reshaped.outer_iter() {
        let probabilities = logits_to_prob(batch.view());

        let mut indexed_values: Vec<(usize, f32)> = probabilities.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_values.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap()); // Sort in descending order

        let top_5: Vec<(usize, f32)> = indexed_values.iter().take(5).cloned().collect();
        top_5_peak_classes.push(top_5);
    }

    Some(top_5_peak_classes)
}

pub fn find_peak_class(output: &ndarray::ArrayD<f32>) -> Option<Vec<usize>> {
    // Calculate total number of elements and the batch size
    let total_elements = output.len();
    let first_dim = output.shape()[0];
    let inferred_dim = total_elements / first_dim;

    // Reshape the tensor
    let reshaped = output.view().into_shape((first_dim, inferred_dim)).ok()?;

    // Find the peak class for each batch
    let mut peak_classes = Vec::with_capacity(first_dim);
    for batch in reshaped.outer_iter() {
        let (index, _) = batch
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;
        peak_classes.push(index);
    }

    Some(peak_classes)
}
