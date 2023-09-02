use crate::onnx_rustime::backend::helper::find_top_5_peak_classes;
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::ops::utils::tensor_proto_to_ndarray;
use crate::onnx_rustime::shared::{DOMAIN_SPECIFIC, IMAGENET_CLASSES, MNIST_CLASSES, VERBOSE};
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
pub fn menu() -> (&'static str, &'static str, &'static str, Option<String>) {
    display_menu();

    let options = vec![
        "AlexNet",
        "CaffeNet",
        "CNN-Mnist",
        "ResNet-152",
        "SqueezeNet",
        "ZFNet",
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

    // Ask if the user wants to save the data
    let save_data_selection = match Select::with_theme(&ColorfulTheme::default())
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

    let default_save_paths = vec![
        "models/bvlcalexnet-12",
        "models/caffenet-12",
        "models/mnist-8",
        "models/resnet152-v2-7",
        "models/squeezenet1.0-12",
        "models/zfnet512-12",
    ];

    let save_path: Option<String> = if save_data_selection {
        let default_path = format!("{}/output_demo.pb", default_save_paths[selection]);
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

    let (model_path, input_path, output_path) = match options[selection] {
        "AlexNet" => (
            "models/bvlcalexnet-12/bvlcalexnet-12.onnx",
            "models/bvlcalexnet-12/test_data_set_0/input_0.pb",
            "models/bvlcalexnet-12/test_data_set_0/output_0.pb",
        ),
        "CaffeNet" => (
            "models/caffenet-12/caffenet-12.onnx",
            "models/caffenet-12/test_data_set_0/input_0.pb",
            "models/caffenet-12/test_data_set_0/output_0.pb",
        ),
        "CNN-Mnist" => {
            {
                let mut d = DOMAIN_SPECIFIC.lock().unwrap();
                *d = true;
            }

            (
                "models/mnist-8/mnist-8.onnx",
                "models/mnist-8/test_data_set_0/input_0.pb",
                "models/mnist-8/test_data_set_0/output_0.pb",
            )
        }
        "ResNet-152" => (
            "models/resnet152-v2-7/resnet152-v2-7.onnx",
            "models/resnet18-v2-7/test_data_set_0/input_0.pb",
            "models/resnet152-v2-7/test_data_set_0/output_0.pb",
        ),
        "SqueezeNet" => (
            "models/squeezenet1.0-12/squeezenet1.0-12.onnx",
            "models/squeezenet1.0-12/test_data_set_0/input_0.pb",
            "models/squeezenet1.0-12/test_data_set_0/output_0.pb",
        ),
        "ZFNet" => (
            "models/zfnet512-12/zfnet512-12.onnx",
            "models/zfnet512-12/test_data_set_0/input_0.pb",
            "models/zfnet512-12/test_data_set_0/output_0.pb",
        ),
        _ => {
            println!("Invalid selection");
            return ("", "", "", None);
        }
    };

    println!("{}", "\n🦀 LOADING MODEL...\n".green().bold());

    (model_path, input_path, output_path, save_path)
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

pub fn display_outputs(predicted: &TensorProto, expected: &TensorProto) {
    let predicted_output = tensor_proto_to_ndarray::<f32>(predicted).unwrap();
    let expected_output = tensor_proto_to_ndarray::<f32>(expected).unwrap();

    println!("{}", "Predicted Output:".bold().magenta());
    println!("{:?}\n", predicted_output);

    let predicted_top_5 = find_top_5_peak_classes(&predicted_output).unwrap();
    println!("{}", "Predicted Top 5 Peak Classes:".bold().magenta());

    let is_domain_specific = {
        let lock = DOMAIN_SPECIFIC.lock().unwrap();
        *lock
    };

    for (batch_index, top_5) in predicted_top_5.iter().enumerate() {
        println!("Batch {}: ", batch_index);
        for &(peak, value) in top_5.iter() {  // Change here
            let class_name = if is_domain_specific {
                MNIST_CLASSES[peak]
            } else {
                IMAGENET_CLASSES[peak]
            };
            println!("Peak: {}, Class: {}, Value: {}", peak, class_name, value);
        }
    }    

    println!("{}", "\nExpected Output:".bold().blue());
    println!("{:?}\n", expected_output);

    let expected_top_5 = find_top_5_peak_classes(&expected_output).unwrap();
    println!("{}", "Expected Top 5 Peak Classes:".bold().blue());

    for (batch_index, top_5) in expected_top_5.iter().enumerate() {
        println!("Batch {}: ", batch_index);
        for &(peak, value) in top_5 {
            let class_name = if is_domain_specific {
                MNIST_CLASSES[peak]
            } else {
                IMAGENET_CLASSES[peak]
            };
            println!("Peak: {}, Class: {}, Value: {}", peak, class_name, value);
        }
    }
    print!("\n");
}


// pub fn display_outputs(predicted: &TensorProto, expected: &TensorProto) {
//     let predicted_output = tensor_proto_to_ndarray::<f32>(predicted).unwrap();
//     let expected_output = tensor_proto_to_ndarray::<f32>(expected).unwrap();

//     println!("{}", "Predicted Output:".bold().magenta());
//     println!("{:?}\n", predicted_output);

//     let predicted_peaks: Vec<usize> = find_peak_class(&predicted_output).unwrap();
//     println!("{}", "Predicted Peak Classes:".bold().magenta(),);

//     let is_domain_specific = {
//         let lock = DOMAIN_SPECIFIC.lock().unwrap();
//         *lock
//     };

//     for &peak in &predicted_peaks {
//         let class_name = if is_domain_specific {
//             MNIST_CLASSES[peak]
//         } else {
//             IMAGENET_CLASSES[peak]
//         };
//         println!("Peak: {}, Class: {}", peak, class_name);
//     }

//     println!("{}", "\nExpected Output:".bold().blue());
//     println!("{:?}\n", expected_output);

//     let expected_peaks = find_peak_class(&expected_output).unwrap();
//     println!("{}", "Expected Peak Classes:".bold().blue());

//     for &peak in &expected_peaks {
//         let class_name = if is_domain_specific {
//             MNIST_CLASSES[peak]
//         } else {
//             IMAGENET_CLASSES[peak]
//         };
//         println!("Peak: {}, Class: {}", peak, class_name);
//     }
//     print!("\n");
// }
