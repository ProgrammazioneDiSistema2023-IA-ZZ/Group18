mod onnx_rustime;
use onnx_rustime::backend::parser::OnnxParser;
use onnx_rustime::backend::pre_processing::serialize_image;
use onnx_rustime::backend::run::run;
use std::env;
mod display;
use display::{display_outputs, menu};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let (model_path, input_path, output_path, save_path_opt) = menu();

    if model_path == "PREPROCESSING" {
        if let Some(save_path) = save_path_opt {
            if let Err(err) = serialize_image(input_path, save_path) {
                eprintln!("Failed to preprocess and serialize image: {:?}", err);
            }
            return;
        } else {
            eprintln!("No save path specified, exiting.");
            return;
        }
    }

    // If not preprocessing, proceed with model loading and inference
    let model = OnnxParser::load_model(model_path).unwrap();

    let input = OnnxParser::load_data(input_path).unwrap();

    let expected_output = if let Some(path) = output_path {
        Some(OnnxParser::load_data(path).unwrap())
    } else {
        None
    };

    // Run the model
    let predicted_output = run(&model, input);

    // If save_path_opt contains a path, save the data
    if let Some(save_path) = save_path_opt {
        OnnxParser::save_data(&predicted_output, save_path).unwrap();
    }

    display_outputs(&predicted_output, expected_output);
}
