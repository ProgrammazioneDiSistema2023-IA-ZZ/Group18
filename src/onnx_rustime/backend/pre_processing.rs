extern crate image;
extern crate ndarray;

use image::{imageops, GenericImageView};
use ndarray::{prelude::*, Array3, ArrayD};

use crate::onnx_rustime::backend::{helper::OnnxError, parser::OnnxParser};
use crate::onnx_rustime::ops::utils::ndarray_to_tensor_proto;

const MIN_SIZE: u32 = 256;
const CROP_SIZE: u32 = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCALE_FACTOR: f32 = 255.0;

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

//fn preprocess_image_mnist(path: &str) -> () {
//    // Load the image
//    let img = image::open(path).unwrap();
//
//    // Convert the RGB image to grayscale
//    let mut grayscale_image = img.grayscale();
//
//    let rescaled_img = grayscale_image.resize(28, 28, imageops::FilterType::Gaussian);
//
//    let inverted_img = rescaled_img.pixels().for_each(|x| x.2 .0[0] -= 255);
//
//    inverted_img
//        .save("data/inverted_grayscale_image.jpg")
//        .unwrap();
//}

use colored::Colorize;

pub fn serialize_image(input_path: String, output_path: String) -> Result<(), OnnxError> {
    println!("{}", "🚀 Starting to preprocess the image...");

    let img_ndarray = preprocess_image(input_path);
    
    println!("{}", "✅ Image preprocessed. Converting to tensor proto...");

    let img_tensorproto = ndarray_to_tensor_proto::<f32>(img_ndarray, "data")?;

    println!("{}", "✅ Tensor proto created. Saving data...");

    let result = OnnxParser::save_data(&img_tensorproto, output_path.clone());

    match result {
        Ok(_) => println!("\n{}\n", format!("🦀 DATA SAVED SUCCESSFULLY TO {}", output_path).magenta().bold()),
        Err(_) => println!("\n{}\n", format!("🛑 Failed to save data to {}", output_path).red().bold()),
    }

    result
}

//#[test]
//fn test_serialize_input() -> Result<(), OnnxError> {
//    // Change the return type to include the error
//    let input_path = "mnist.jpg";
//    let output_path = "data/test_serialized_data.pb";
//    // Perform serialization
//    preprocess_image_mnist(input_path);
//    Ok(())
//}

//#[cfg(test)]
//mod tests {
//
//    const MIN_SIZE: u32 = 256;
//    const CROP_SIZE: u32 = 224;
//    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
//    const STD: [f32; 3] = [0.229, 0.224, 0.225];
//    const SCALE_FACTOR: f32 = 255.0;
//
//    use super::{preprocess_image, serialize_input};
//    use crate::{
//        onnx_rustime::backend::{helper::OnnxError, parser::OnnxParser},
//        onnx_rustime::ops::utils::tensor_proto_to_ndarray,
//    };
//    use image::*;
//    use ndarray::*;
//
//    #[test]
//    fn test_original_input() -> () {
//        let tensorproto =
//            OnnxParser::load_data("models/resnet152-v2-7/test_data_set_0/input_0.pb").unwrap();
//        let ndarray = tensor_proto_to_ndarray::<f32>(&tensorproto).unwrap();
//
//        let mut ndarray = ndarray.to_shape([3, 224, 224]).unwrap();
//
//        let mean = Array::from_shape_vec((3, 1, 1), vec![MEAN[0], MEAN[1], MEAN[2]]).unwrap();
//
//        let std = Array::from_shape_vec((3, 1, 1), vec![STD[0], STD[1], STD[2]]).unwrap();
//
//        // // Reverse normalization
//        for i in 0..3 {
//            let mean_value = mean[[i, 0, 0]];
//            let std_value = std[[i, 0, 0]];
//
//            let mut slice = ndarray.slice_mut(s![i, .., ..]);
//            slice.map_inplace(|x| *x = ((*x + mean_value) * std_value) * 255.0);
//        }
//
//        //ndarray *= &std; // multiply by standard deviation
//        //ndarray += &mean; // add back the mean
//
//        // Convert Array3<f32> to Array3<u8>
//        let mut arr_u8 = ndarray.mapv(|x| x.round() as u8); // round and convert to u8
//                                                            //println!("is it black? {:?}", arr_u8);
//
//        // Swap axes back to HWC layout
//        arr_u8.swap_axes(0, 2);
//
//        // Convert ndarray to image data (assuming `arr_u8` is in HWC format)
//        let img_buffer =
//            RgbImage::from_raw(CROP_SIZE as u32, CROP_SIZE as u32, arr_u8.into_raw_vec()).unwrap();
//
//        // Save the image
//        img_buffer.save("reconstructed_image.jpg").unwrap();
//    }
//
//    #[test]
//    fn test_serialize_input() -> Result<(), OnnxError> {
//        // Change the return type to include the error
//        let input_path = "data/imagenet-sample-images/n01784675_centipede.JPEG";
//        let output_path = "data/test_serialized_data.pb";
//
//        // Perform serialization
//        serialize_input(input_path, output_path)?;
//
//        Ok(())
//    }
//
//    #[test]
//    fn test_preprocess_image() {
//        let processed_image: ArrayD<f32> =
//            preprocess_image("data/imagenet-sample-images/n01514668_cock.JPEG");
//
//        // Check dimensions: [1, 3, 224, 224]
//        assert_eq!(processed_image.shape(), &[1, 3, 224, 224]);
//    }
//}
