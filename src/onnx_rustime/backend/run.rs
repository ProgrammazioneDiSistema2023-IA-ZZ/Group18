use crate::onnx_rustime::backend::helper::OnnxError;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{ModelProto, NodeProto, TensorProto};
use crate::onnx_rustime::ops::*;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::time::Instant;

use crate::onnx_rustime::shared::VERBOSE;

/// Executes a given ONNX model using the provided input tensor and returns the output tensor.
///
/// This function processes the graph nodes in the order they appear in the model's graph definition.
/// It also handles initializers for nodes and routes the output of one node as the input for subsequent nodes.
///
/// # Arguments
///
/// * `model` - The ONNX model to be executed.
/// * `input_tensor` - The input tensor for the model.
///
/// # Returns
///
/// * `TensorProto` - The output tensor after model execution.
pub fn run(model: &ModelProto, input_tensor: TensorProto) -> TensorProto {
    // Capture the current time before running the model
    let start = Instant::now();

    // Extract the graph from the model.
    let graph = model.get_graph();

    // Initialize a map to hold the tensors for each node's input.
    let mut input_map: HashMap<String, TensorProto> = HashMap::new();
    input_map.insert(graph.input[0].name.clone(), input_tensor);

    // Map the initializers by their names for easy lookup.
    let initializers_map: HashMap<String, TensorProto> = graph
        .get_initializer()
        .iter()
        .map(|tensor_proto| (tensor_proto.name.clone(), tensor_proto.clone()))
        .collect();

    // Initialize a progress bar
    let bar = ProgressBar::new(graph.get_node().len() as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("\n{prefix:.bold.blue} {bar:40.blue/blue} [{pos}/{len} nodes]")
            .unwrap()
            .progress_chars("█▁"),
    );

    // Iterate over each node in the graph.
    for node in &graph.node {
        // Gather the inputs for the current node.
        let node_inputs: Vec<_> = node
            .get_input()
            .iter()
            .filter_map(|name| input_map.get(name))
            .collect();

        // Gather the initializers for the current node.
        let node_initializers: Vec<_> = node
            .get_input()
            .iter()
            .filter_map(|name| initializers_map.get(name))
            .collect();

        bar.println(format!(
            "{} {} {}",
            "🚀 Running Node:".bold(),
            node.get_op_type(),
            node.get_name()
        ));

        let output_tensor = if *VERBOSE.lock().unwrap() {
            run_node_verbose(&bar, node, &node_inputs, &node_initializers)
        } else {
            run_node(node, &node_inputs, &node_initializers).expect("Failed to run node")
        };

        let output_name = output_tensor.get_name().to_string();
        // Store the output tensor so it can be used as input for subsequent nodes.
        input_map.insert(output_name, output_tensor);

        // Increment the progress bar
        bar.inc(1);
    }
    bar.finish();

    let duration = start.elapsed();
    println!("\n\n{} ({:?})\n", "🦀 SUCCESSFULLY RUN NETWORK!".bold().magenta(), duration);

    // Return the output tensor for the entire model.
    input_map
        .get(&graph.get_output()[0].name)
        .expect("Output tensor not found")
        .clone()
}

/// Executes a specific node in the ONNX graph.
///
/// This function maps the node's operation type (e.g., "Conv", "Add", etc.) to its corresponding
/// execution function and passes the required inputs and initializers.
///
/// # Arguments
///
/// * `node` - The node to be executed.
/// * `inputs` - A list of input tensors for the node.
/// * `initializers` - A list of initializer tensors for the node.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - The output tensor from the node's execution or an error.
fn run_node(
    node: &NodeProto,
    inputs: &Vec<&TensorProto>,
    initializers: &Vec<&TensorProto>,
) -> Result<TensorProto, OnnxError> {
    match node.get_op_type() {
        "Add" => add(inputs, Some(initializers), node),
        "BatchNormalization" => batch_normalization(inputs[0], initializers, node),
        "Concat" => concat(inputs, node),
        "Conv" => conv(inputs[0], initializers, node),
        "Dropout" => dropout(inputs[0], Some(initializers), node),
        "Exp" => exp(inputs[0], node),
        "Flatten" => flatten(inputs[0], node),
        "Gemm" => gemm(inputs, Some(initializers), node),
        "GlobalAveragePool" => global_average_pool(inputs[0], node),
        "LRN" => lrn(inputs[0], node),
        "MatMul" => matmul(inputs, Some(initializers), node),
        "MaxPool" => maxpool(inputs[0], node),
        "ReduceSum" => reduce_sum(inputs[0], node),
        "Relu" => relu(inputs[0], node),
        "Reshape" => reshape(inputs.get(0).copied(), initializers, node),
        "Softmax" => softmax(inputs[0], node),
        _ => Err(OnnxError::InternalError(format!(
            "Operation '{}' not found!",
            node.get_op_type()
        ))),
    }
}

fn truncate_with_ellipsis(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("…{}", &s[s.len() - (max_len - 1)..])
    } else {
        s.to_string()
    }
}

fn run_node_verbose(
    bar: &ProgressBar,
    node: &NodeProto,
    input_tensors: &Vec<&TensorProto>,
    initializer_tensors: &Vec<&TensorProto>,
) -> TensorProto {
    let name_column_width = 35; // Fixed width

    if !input_tensors.is_empty() || !initializer_tensors.is_empty() {
        bar.println(format!(
            "{:<16} {:<width$} {}",
            "Operand".red(),
            "Name".red(),
            "Shape".red(),
            width = name_column_width
        ));
        bar.println(format!(
            "{} {} {}",
            "---------------".red(),
            "-".repeat(name_column_width).red(),
            "------------------".red()
        ));

        for input in input_tensors {
            bar.println(format!(
                "{:<15} {:<width$} {:?}",
                "🟢 Input".bright_green(),
                truncate_with_ellipsis(input.get_name(), name_column_width),
                input.get_dims(),
                width = name_column_width
            ));
        }

        for initializer in initializer_tensors {
            bar.println(format!(
                "{:<15} {:<width$} {:?}",
                "🟡 Initializer".bright_yellow(),
                truncate_with_ellipsis(initializer.get_name(), name_column_width),
                initializer.get_dims(),
                width = name_column_width
            ));
        }
    }

    let output_tensor =
        run_node(node, input_tensors, initializer_tensors).expect("Failed to run node");

    bar.println(format!(
        "{:<15} {:<width$} {:?}\n\n",
        "🟣 Output".bright_purple(),
        truncate_with_ellipsis(output_tensor.get_name(), name_column_width),
        output_tensor.get_dims(),
        width = name_column_width
    ));

    output_tensor
}
