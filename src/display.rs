use crate::onnx_rustime::backend::helper::find_top_5_peak_classes;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::tensor_proto_to_ndarray;
use crate::onnx_rustime::shared::{Model, IMAGENET_CLASSES, MNIST_CLASSES, MODEL_NAME, VERBOSE};
use colored::*;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use std::path::Path;
use std::process;

const RUST_COLOR: &[u8] = &[209, 114, 119];

/// Display the main menu and return the user's selected model, input path, output path, and optional save path.
///
/// The function will:
/// 1. Display the main menu.
/// 2. Ask the user to select a network.
/// 3. Ask if the user wants to save the output data.
/// 4. Ask for the path to save the output data.
/// 5. Ask if the user wants to run in verbose mode.
///
/// Returns a tuple containing:
/// - model_path: Path to the selected ONNX model.
/// - input_path: Path to the input test data for the selected model.
/// - ground_truth_output_path: Path to the expected output test data for the selected model.
/// - save_path: Optional path where the user wants to save the output data.
pub fn menu() -> (String, String, Option<String>, Option<String>) {
    display_menu();

    let options = vec![
        "AlexNet",
        "CaffeNet",
        "CNN-Mnist",
        "ResNet-152",
        "SqueezeNet",
        "ZFNet",
        "Pre-process and serialize an image",
        "Exit",
    ];

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a network to run")
        .items(&options)
        .default(0)
        .interact()
        .unwrap();

    if selection == options.len() - 1 {
        println!("Exiting...");
        process::exit(0);
    } else {
        println!("You selected: {}", options[selection]);
    }

    let default_input_paths = vec![
        "models/bvlcalexnet-12/test_data_set_0/input_0.pb",
        "models/caffenet-12/test_data_set_0/input_0.pb",
        "models/mnist-8/test_data_set_0/input_0.pb",
        "models/resnet18-v2-7/test_data_set_0/input_0.pb",
        "models/squeezenet1.0-12/test_data_set_0/input_0.pb",
        "models/zfnet512-12/test_data_set_0/input_0.pb",
    ];

    let input_path: String = loop {
        if options[selection] == "Pre-process and serialize an image" {
            // Special prompt logic for image processing
            let path: String = Input::with_theme(&ColorfulTheme::default())
              .with_prompt("Please provide a path for the image to be processed:\n(type 'BACK' to go back)")
              .interact()
              .unwrap();

            if path.trim().to_uppercase() == "BACK" {
                clear_screen();
                return menu();
            }

            if Path::new(&path).exists() {
                break path;
            } else {
                println!(
                    "{}",
                    "Provided path does not exist.\nPlease provide a valid image path.".red()
                );
            }
        } else {
            // Default prompt logic for models
            let default_path = default_input_paths[selection].to_string();
            let path: String = Input::with_theme(&ColorfulTheme::default())
              .with_prompt("Please provide a path for the input data:\n(type 'BACK' to go back, Enter for default)")
              .default(default_path.clone())
              .interact()
              .unwrap();

            if path.trim().to_uppercase() == "BACK" {
                clear_screen();
                return menu();
            }

            if Path::new(&path).exists() {
                break path;
            } else {
                println!("{}", "Provided path does not exist.\nPlease provide a valid path or press Enter for default.".red());
            }
        }
    };

    let mut save_data_selection = true; // Default to true for "Pre-process and serialize an image"

    if options[selection] != "Pre-process and serialize an image" {
        // Ask only if the user didn't select "Pre-process and serialize an image"
        save_data_selection = match Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Save the output data?")
            .items(&["Yes", "No", "Back"])
            .default(0)
            .interact()
            .unwrap()
        {
            0 => true,
            1 => false,
            2 => {
                clear_screen();
                return menu();
            }
            _ => false,
        };
    }

    let default_save_paths = vec![
        "models/bvlcalexnet-12",
        "models/caffenet-12",
        "models/mnist-8",
        "models/resnet152-v2-7",
        "models/squeezenet1.0-12",
        "models/zfnet512-12",
        ".",
    ];

    let save_path: Option<String> = if save_data_selection {
        let default_path = if options[selection] == "Pre-process and serialize an image" {
            // Derive the save path from the input image path
            let stem = Path::new(&input_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("pre_processed_input");
            format!("./{}_pre_processed.pb", stem)
        } else {
            format!("{}/output_demo.pb", default_save_paths[selection])
        };

        loop {
            let mut path: String = Input::with_theme(&ColorfulTheme::default())
              .with_prompt("Please provide a path to save output:\n(type 'BACK' to go back, Enter for default)")
              .default(default_path.clone())
              .interact()
              .unwrap();

            if path.trim().to_uppercase() == "BACK" {
                clear_screen();
                return menu();
            }

            // Append .pb extension if not present
            if !path.ends_with(".pb") {
                path.push_str(".pb");
            }

            // Check if parent directory of the path exists
            if let Some(parent) = Path::new(&path).parent() {
                if parent.exists() {
                    break Some(path);
                } else {
                    println!("{}", "Parent directory of the provided path does not exist.\nPlease provide a valid path or press Enter for default.".red());
                }
            } else {
                println!("Please provide a valid path or press Enter for default.");
            }
        }
    } else {
        None
    };

    if options[selection] != "Pre-process and serialize an image" {
        // Ask if the user wants to run in verbose mode
        let verbose_selection = match Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Run in verbose mode?")
            .items(&["Yes", "No", "Back"])
            .default(0)
            .interact()
            .unwrap()
        {
            0 => true,
            1 => false,
            2 => {
                clear_screen();
                return menu();
            }
            _ => false,
        };

        {
            let mut v = VERBOSE.lock().unwrap();
            *v = verbose_selection;
        }
    }

    let (model_path, output_path) = match options[selection] {
        "AlexNet" => {
            {
                let mut d = MODEL_NAME.lock().unwrap();
                *d = Model::AlexNet;
            }
            (
                "models/bvlcalexnet-12/bvlcalexnet-12.onnx",
                if input_path == default_input_paths[selection] {
                    Some("models/bvlcalexnet-12/test_data_set_0/output_0.pb".to_string())
                } else {
                    None
                },
            )
        }
        "CaffeNet" => {
            {
                let mut d = MODEL_NAME.lock().unwrap();
                *d = Model::CaffeNet;
            }
            (
                "models/caffenet-12/caffenet-12.onnx",
                if input_path == default_input_paths[selection] {
                    Some("models/caffenet-12/test_data_set_0/output_0.pb".to_string())
                } else {
                    None
                },
            )
        }
        "CNN-Mnist" => {
            {
                let mut d = MODEL_NAME.lock().unwrap();
                *d = Model::Mnist;
            }
            (
                "models/mnist-8/mnist-8.onnx",
                if input_path == default_input_paths[selection] {
                    Some("models/mnist-8/test_data_set_0/output_0.pb".to_string())
                } else {
                    None
                },
            )
        }
        "ResNet-152" => {
            {
                let mut d = MODEL_NAME.lock().unwrap();
                *d = Model::ResNet;
            }
            (
                "models/resnet152-v2-7/resnet152-v2-7.onnx",
                if input_path == default_input_paths[selection] {
                    Some("models/resnet152-v2-7/test_data_set_0/output_0.pb".to_string())
                } else {
                    None
                },
            )
        }
        "SqueezeNet" => {
            {
                let mut d = MODEL_NAME.lock().unwrap();
                *d = Model::SqueezeNet;
            }
            (
                "models/squeezenet1.0-12/squeezenet1.0-12.onnx",
                if input_path == default_input_paths[selection] {
                    Some("models/squeezenet1.0-12/test_data_set_0/output_0.pb".to_string())
                } else {
                    None
                },
            )
        }
        "ZFNet" => {
            {
                let mut d = MODEL_NAME.lock().unwrap();
                *d = Model::ZFNet;
            }
            (
                "models/zfnet512-12/zfnet512-12.onnx",
                if input_path == default_input_paths[selection] {
                    Some("models/zfnet512-12/test_data_set_0/output_0.pb".to_string())
                } else {
                    None
                },
            )
        }
        "Pre-process and serialize an image" => {
            {
                let mut d = MODEL_NAME.lock().unwrap();
                *d = Model::PreProcessing;
            }
            ("PREPROCESSING", None)
        }
        _ => {
            println!("Invalid selection");
            ("", None)
        }
    };

    println!("{}", "\n🦀 LOADING...\n".green().bold());

    (model_path.to_string(), input_path, output_path, save_path)
}

fn display_menu() {
    let onnx_art = r#"   
    ____  _   _ _   ___   __   _____           _   _                
   / __ \| \ | | \ | \ \ / /  |  __ \         | | (_)               
  | |  | |  \| |  \| |\ V /   | |__) |   _ ___| |_ _ _ __ ___   ___ 
  | |  | | . ` | . ` | > <    |  _  / | | / __| __| | '_ ` _ \ / _ \
  | |__| | |\  | |\  |/ . \   | | \ \ |_| \__ \ |_| | | | | | |  __/
   \____/|_| \_|_| \_/_/ \_\  |_|  \_\__,_|___/\__|_|_| |_| |_|\___|"#;

    let _separator: &str = r#"
===============================================
"#;

    // Clear the screen
    clear_screen();

    // Print the ASCII art with colors
    println!(
        "{}",
        onnx_art
            .blink()
            .truecolor(RUST_COLOR[0], RUST_COLOR[1], RUST_COLOR[2])
    );

    let info = r#"
+--------------------------------------------------------------------+
| ONNX Rustime - The Rustic ONNX Experience                          |
|                                                                    |
| - 🦀 Rust-inspired ONNX runtime.                                   |
| - 📖 Robust parser for ONNX files & test data.                     |
| - 🚀 Run network inference post-parsing.                           |
| - 🔨 Scaffold for adding operations.                               |
| - 📊 Demo-ready with multiple CNNs. Extend freely!                 |
| - 🔄 Supports batching for simultaneous inferences.                |
| - 💾 Serialize models & data with ease.                            |
| - 🐍 Seamless Python integration via Rust bindings.                |
| - 🟢 Seamless JavaScript integration via Rust bindings.            |
+--------------------------------------------------------------------+
"#;

    println!(
        "{}",
        info.truecolor(RUST_COLOR[0], RUST_COLOR[1], RUST_COLOR[2])
    );
}

fn clear_screen() {
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
}


fn truncate_with_ellipsis(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("…{}", &s[s.len() - (max_len - 1)..])
    } else {
        s.to_string()
    }
}

pub fn display_outputs(predicted: &TensorProto, expected: Option<TensorProto>) {
    let name_column_width = 35; // Fixed width
    let model_name = {
        let lock = MODEL_NAME.lock().unwrap();
        lock.clone()
    };

    let predicted_output = tensor_proto_to_ndarray::<f32>(predicted).unwrap();

    println!("{}", "Predicted Output:".bold().magenta());
    println!("{:?}\n", predicted_output);

    let predicted_top_5 = find_top_5_peak_classes(&predicted_output).unwrap();
    for (batch_index, top_5) in predicted_top_5.iter().enumerate() {
        println!("[Batch {}]\n", batch_index);
        println!(
            "{:<16} {:<width$} {}",
            "Predicted Peak".bold().magenta(),
            "Class Name".bold().magenta(),
            "Percentage".bold().magenta(),
            width = name_column_width
        );
        println!(
            "{} {} {}",
            "---------------".bold().magenta(),
            "-".repeat(name_column_width).bold().magenta(),
            "------------------".bold().magenta()
        );
        for &(peak, value) in top_5.iter() {
            let class_name = match model_name {
                Model::Mnist => MNIST_CLASSES[peak],
                _ => IMAGENET_CLASSES[peak],
            };
            let truncated_class_name = truncate_with_ellipsis(class_name, name_column_width);
            let percentage = value * 100.0;
            println!(
                "{:<16} {:<width$} {:.2}%",
                peak,
                truncated_class_name,
                percentage,
                width = name_column_width
            );
        }
    }

    if let Some(expected_tensor) = expected {
        println!("{}", "\nExpected Output:".bold().blue());

        let expected_output = tensor_proto_to_ndarray::<f32>(&expected_tensor).unwrap();
        println!("{:?}\n", expected_output);

        let expected_top_5 = find_top_5_peak_classes(&expected_output).unwrap();
        for (batch_index, top_5) in expected_top_5.iter().enumerate() {
            println!("[Batch {}]\n", batch_index);
            println!(
                "{:<16} {:<width$} {}",
                "Expected Peak".bold().blue(),
                "Class Name".bold().blue(),
                "Percentage".bold().blue(),
                width = name_column_width
            );
            println!(
                "{} {} {}",
                "---------------".bold().blue(),
                "-".repeat(name_column_width).bold().blue(),
                "------------------".bold().blue()
            );

            for &(peak, value) in top_5.iter() {
                let class_name = match model_name {
                    Model::Mnist => MNIST_CLASSES[peak],
                    _ => IMAGENET_CLASSES[peak],
                };
                let truncated_class_name = truncate_with_ellipsis(class_name, name_column_width);
                let percentage = value * 100.0;
                println!(
                    "{:<16} {:<width$} {:.2}%",
                    peak,
                    truncated_class_name,
                    percentage,
                    width = name_column_width
                );
            }
        }
    }
    println!("\n");
}
