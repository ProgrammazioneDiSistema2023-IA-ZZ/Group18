extern crate image;
extern crate ndarray;

use colored::Colorize;

use image::{imageops, GenericImageView};
use ndarray::{prelude::*, Array3, ArrayD};

use crate::onnx_rustime::backend::{helper::OnnxError, parser::OnnxParser};
use crate::onnx_rustime::ops::utils::ndarray_to_tensor_proto;

const MIN_SIZE: u32 = 256;
const CROP_SIZE: u32 = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCALE_FACTOR: f32 = 255.0;

/// `preprocess_image` - Preprocesses an image for deep learning model input.
///
/// This function takes an image file path, loads the image, resizes it, performs cropping,
/// converts it to an ndarray with proper shape, normalizes the pixel values, and adds a batch dimension.
///
/// # Arguments
///
/// * `path` - A String containing the file path to the image.
///
/// # Returns
///
/// * `ArrayD<f32>` - Returns the preprocessed image as an ArrayD with shape (1, 3, CROP_SIZE, CROP_SIZE).
///
/// # Example
///
/// ```rust
/// let image_array = preprocess_image("/path/to/image.jpg".to_string());
/// ```
fn preprocess_image(path: String) -> ArrayD<f32> {
    // Load the image
    let mut img = image::open(path).unwrap();

    let (width, height) = img.dimensions();

    // Resize the image with a minimum size of MIN_SIZE while maintaining the aspect ratio
    let (nwidth, nheight) = if width > height {
        (MIN_SIZE * width / height, MIN_SIZE)
    } else {
        (MIN_SIZE, MIN_SIZE * height / width)
    };

    img = img.resize(nwidth, nheight, imageops::FilterType::Gaussian);

    // Crop the image to CROP_SIZE from the center
    let crop_x = (nwidth - CROP_SIZE) / 2;
    let crop_y = (nheight - CROP_SIZE) / 2;

    img = img.crop_imm(crop_x, crop_y, CROP_SIZE, CROP_SIZE);

    // Convert the image to RGB and transform it into ndarray
    // this is an ImageBuffer with RGB values ranging from 0 to 255
    let img_rgb = img.to_rgb8();

    let raw_data = img_rgb.into_raw();

    let (mut rs, mut gs, mut bs) = (Vec::new(), Vec::new(), Vec::new());

    for i in 0..raw_data.len() / 3 {
        rs.push(raw_data[3 * i]);
        gs.push(raw_data[3 * i + 1]);
        bs.push(raw_data[3 * i + 2]);
    }

    let r_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), rs).unwrap();
    let g_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), gs).unwrap();
    let b_array: Array2<u8> =
        Array::from_shape_vec((CROP_SIZE as usize, CROP_SIZE as usize), bs).unwrap();

    // Stack them to make an Array3
    let mut arr: Array3<u8> =
        ndarray::stack(Axis(2), &[r_array.view(), g_array.view(), b_array.view()]).unwrap();
    // Transpose it from HWC to CHW layout
    arr.swap_axes(0, 2);

    let mean = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            MEAN[0] * SCALE_FACTOR,
            MEAN[1] * SCALE_FACTOR,
            MEAN[2] * SCALE_FACTOR,
        ],
    )
    .unwrap();

    let std = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            STD[0] * SCALE_FACTOR,
            STD[1] * SCALE_FACTOR,
            STD[2] * SCALE_FACTOR,
        ],
    )
    .unwrap();

    let mut arr_f: Array3<f32> = arr.mapv(|x| x as f32);

    arr_f -= &mean;
    arr_f /= &std;

    // Add a batch dimension, shape becomes (1, 3, CROP_SIZE, CROP_SIZE)
    let arr_f_batch: Array4<f32> = arr_f.insert_axis(Axis(0));

    // Convert Array4 to ArrayD
    let arr_d: ArrayD<f32> = arr_f_batch.into_dimensionality().unwrap();

    arr_d
}

/// `serialize_image` - Preprocesses and serializes an image, saving it to a file.
///
/// This function preprocesses an image and converts it to a tensor proto, which is then saved to a file.
///
/// # Arguments
///
/// * `input_path` - A String containing the file path to the input image.
/// * `output_path` - A String containing the file path where the serialized data will be saved.
///
/// # Returns
///
/// * `Result<(), OnnxError>` - Returns Ok(()) if the serialization is successful, or an OnnxError if it fails.
///
/// # Example
///
/// ```rust
/// let result = serialize_image("/input/image.jpg".to_string(), "/output/data.pb".to_string());
/// ```
pub fn serialize_image(input_path: String, output_path: String) -> Result<(), OnnxError> {
    println!("{}", "🚀 Starting to preprocess the image...");

    let img_ndarray = preprocess_image(input_path);

    println!("{}", "✅ Image preprocessed. Converting to tensor proto...");

    let img_tensorproto = ndarray_to_tensor_proto::<f32>(img_ndarray, "data")?;

    println!("{}", "✅ Tensor proto created. Saving data...");

    let result = OnnxParser::save_data(&img_tensorproto, output_path.clone());

    match result {
        Ok(_) => println!(
            "\n{}\n",
            format!("🦀 DATA SAVED SUCCESSFULLY TO {}", output_path)
                .magenta()
                .bold()
        ),
        Err(_) => println!(
            "\n{}\n",
            format!("🛑 Failed to save data to {}", output_path)
                .red()
                .bold()
        ),
    }

    result
}
