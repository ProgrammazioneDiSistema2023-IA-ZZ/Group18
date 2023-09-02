mod onnx_rustime;
use onnx_rustime::backend::parser::OnnxParser;
use onnx_rustime::backend::run::run;
use onnx_rustime::ops::utils::tensor_proto_to_ndarray;
use std::env;
mod display;
use display::{display_outputs, menu};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (model_path, input_path, output_path, save_path_opt) = menu();

    let model = OnnxParser::load_model(model_path).unwrap();
    let input = OnnxParser::load_data(input_path).unwrap();
    let expected_output = OnnxParser::load_data(output_path).unwrap();

    println!("input to the net: {:?}", tensor_proto_to_ndarray::<f32>(&input));

    // Run the model
    let predicted_output = run(&model, input);

    // If save_path_opt contains a path, save the data
    if let Some(save_path) = save_path_opt {
        OnnxParser::save_data(&predicted_output, &save_path).unwrap();
    }

    display_outputs(&predicted_output, &expected_output);

}
