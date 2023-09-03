use crate::onnx_rustime::backend::helper::{Attribute, OnnxError};
use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::{NodeProto, TensorProto};
use crate::onnx_rustime::ops::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, get_ints_attribute,
    get_string_attribute, pad_matrix_3d, stack_along_batch_dimension, tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
use num_traits::Float;
use rayon::prelude::*;

pub type DataRepresentation<F> = Array3<F>;

/// Padding (specific way of adding zeros to the input matrix) kind used in the convolution.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Padding {
    /// Output has the same shape as input.
    Same,
    /// Padding is only used to make input fit the kernel.
    Valid,
}

/// Rust implementation of a convolutional layer.
/// The weight matrix shall have dimension (in that order)
/// (input channels, output channels, kernel width, kernel height),
/// to comply with the order in which pytorch weights are saved.
struct ConvolutionLayer<F: Float> {
    /// Weight matrix of the kernel
    kernel: Array4<F>,
    bias: Option<Array1<F>>,
    stride: usize,
    padding: Padding,
}

impl<F: 'static + Float + std::ops::AddAssign> ConvolutionLayer<F> {
    /// Creates new convolution layer.
    /// The weights are given in Pytorch layout.
    /// (out channels, in channels, kernel height, kernel width)
    /// Bias: (output height * output width, 1)
    pub fn new(
        weights: Array4<F>,
        bias_array: Option<Array1<F>>,
        stride: usize,
        padding: Padding,
    ) -> ConvolutionLayer<F> {
        assert!(stride > 0, "Stride of 0 passed");
        ConvolutionLayer {
            kernel: weights,
            bias: bias_array,
            stride,
            padding,
        }
    }

    /// Analog to conv2d.
    pub fn convolve(&self, image: &DataRepresentation<F>) -> DataRepresentation<F> {
        conv2d(
            &self.kernel,
            self.bias.as_ref(),
            image,
            self.padding,
            self.stride,
        )
    }
}

fn get_padding_size(
    input_h: usize,
    input_w: usize,
    stride: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    let pad_along_height: usize;
    let pad_along_width: usize;
    let idx_0: usize = 0;

    if input_h % stride == idx_0 {
        pad_along_height = (kernel_h - stride).max(idx_0);
    } else {
        pad_along_height = (kernel_h - (input_h % stride)).max(idx_0);
    };
    if input_w % stride == idx_0 {
        pad_along_width = (kernel_w - stride).max(idx_0);
    } else {
        pad_along_width = (kernel_w - (input_w % stride)).max(idx_0);
    };

    let pad_top = pad_along_height / 2;
    let pad_bottom = pad_along_height - pad_top;
    let pad_left = pad_along_width / 2;
    let pad_right = pad_along_width - pad_left;

    (
        pad_along_height,
        pad_along_width,
        pad_bottom,
        pad_top,
        pad_right,
        pad_left,
    )
}

fn im2col_ref<'a, T, F: 'a + Float>(
    im_arr: T,
    ker_height: usize,
    ker_width: usize,
    im_height: usize,
    im_width: usize,
    im_channel: usize,
    stride: usize,
) -> Array2<F>
where
    // Args:
    //   im_arr: image matrix to be translated into columns, (C,H,W)
    //   ker_height: filter height (hh)
    //   ker_width: filter width (ww)
    //   im_height: image height
    //   im_width: image width
    //
    // Returns:
    //   col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
    //         new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    T: AsArray<'a, F, Ix3>,
{
    let im2d_arr: ArrayView3<F> = im_arr.into();
    let new_h = (im_height - ker_height) / stride + 1;
    let new_w = (im_width - ker_width) / stride + 1;
    let mut cols_img: Array2<F> =
        Array::zeros((new_h * new_w, im_channel * ker_height * ker_width));
    let mut cont = 0_usize;
    for i in 1..new_h + 1 {
        for j in 1..new_w + 1 {
            let patch = im2d_arr.slice(s![
                ..,
                (i - 1) * stride..((i - 1) * stride + ker_height),
                (j - 1) * stride..((j - 1) * stride + ker_width),
            ]);
            let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

            cols_img.row_mut(cont).assign(&patchrow_unwrap);
            cont += 1;
        }
    }
    cols_img
}

/// Performs a convolution on the given image data using this layers initializers.
/// We always convolve on flattened images and expect the input array in im2col
/// style format.
///
/// Input:
/// -----------------------------------------------
/// - kernel_weights: weights of shape (F, C, HH, WW)
/// - im2d: Input data of shape (C, H, W)
/// -----------------------------------------------
/// - 'stride': The number of pixels between adjacent receptive fields in the
///     horizontal and vertical directions, must be int
/// - 'pad': "Same" or "Valid"

/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (F, H', W')
fn conv2d<'a, T, V, F: 'static + Float + std::ops::AddAssign>(
    kernel_weights: T,
    bias: Option<&Array1<F>>,
    im2d: V,
    padding: Padding,
    stride: usize,
) -> DataRepresentation<F>
where
    // This trait bound ensures that kernel and im2d can be passed as owned array or view.
    // AsArray just ensures that im2d can be converted to an array view via ".into()".
    V: AsArray<'a, F, Ix3>,
    T: AsArray<'a, F, Ix4>,
{
    // Initialisations
    let im2d_arr: ArrayView3<F> = im2d.into();
    let kernel_weights_arr: ArrayView4<F> = kernel_weights.into();
    let im_col: Array2<F>; // output of fn: im2col_ref()
    let new_im_height: usize;
    let new_im_width: usize;
    let weight_shape = kernel_weights_arr.shape();
    let num_filters = weight_shape[0] as usize;
    let num_channels_out = weight_shape[1] as usize;
    let kernel_height = weight_shape[2] as usize;
    let kernel_width = weight_shape[3] as usize;

    // Dimensions: C, H, W
    let im_channel = im2d_arr.len_of(Axis(0));
    let im_height = im2d_arr.len_of(Axis(1));
    let im_width = im2d_arr.len_of(Axis(2));

    // Calculate output shapes H', W' for two types of Padding
    if padding == Padding::Same {
        // H' = H / stride
        // W' = W / stride

        let h_float = im_height as f32;
        let w_float = im_width as f32;
        let stride_float = stride as f32;

        let new_im_height_float = (h_float / stride_float).ceil();
        let new_im_width_float = (w_float / stride_float).ceil();

        new_im_height = new_im_height_float as usize;
        new_im_width = new_im_width_float as usize;
    } else {
        // H' =  ((H - HH) / stride ) + 1
        // W' =  ((W - WW) / stride ) + 1
        new_im_height = ((im_height - kernel_height) / stride) + 1;
        new_im_width = ((im_width - kernel_width) / stride) + 1;
    };

    // weights.reshape(F, HH*WW*C)
    let filter_col = kernel_weights_arr
        .into_shape((num_filters, kernel_height * kernel_width * num_channels_out))
        .unwrap();

    // fn:im2col() for different Paddings
    if padding == Padding::Same {
        let (pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right) =
            get_padding_size(im_height, im_width, stride, kernel_height, kernel_width);
        let mut im2d_arr_pad: Array3<F> = Array::zeros((
            num_channels_out,
            im_height + pad_num_h,
            im_width + pad_num_w,
        ));
        let pad_bottom_int = (im_height + pad_num_h) - pad_bottom;
        let pad_right_int = (im_width + pad_num_w) - pad_right;

        im2d_arr_pad
            .slice_mut(s![.., pad_top..pad_bottom_int, pad_left..pad_right_int])
            .assign(&im2d_arr);

        let im_height_pad = im2d_arr_pad.len_of(Axis(1));
        let im_width_pad = im2d_arr_pad.len_of(Axis(2));

        im_col = im2col_ref(
            im2d_arr_pad.view(),
            kernel_height,
            kernel_width,
            im_height_pad,
            im_width_pad,
            im_channel,
            stride,
        );
    } else {
        im_col = im2col_ref(
            im2d_arr,
            kernel_height,
            kernel_width,
            im_height,
            im_width,
            im_channel,
            stride,
        );
    };
    let filter_transpose = filter_col.t();

    let mul = im_col.dot(&filter_transpose);
    let output = mul
        .into_shape((new_im_height, new_im_width, num_filters))
        .unwrap()
        .permuted_axes([2, 0, 1]);

    add_bias(&output, bias)
}

fn add_bias<F>(x: &Array3<F>, bias: Option<&Array1<F>>) -> Array3<F>
where
    F: 'static + Float + std::ops::AddAssign,
{
    if let Some(bias_array) = bias {
        assert!(
            bias_array.shape()[0] == x.shape()[0],
            "Bias array has the wrong shape {:?} for vec of shape {:?}",
            bias_array.shape(),
            x.shape()
        );

        (x + &bias_array
            .clone()
            .insert_axis(Axis(1))
            .insert_axis(Axis(2))
            .broadcast(x.shape())
            .unwrap())
            .into_dimensionality()
            .unwrap()
    } else {
        x.clone()
    }
}

/// Determines the padding type and processes the input based on the `auto_pad` attribute.
///
/// This function processes the input tensor to determine the padding type and prepares the
/// input data for the convolution operation. If the `auto_pad` attribute is set to "NOT_SET",
/// the input tensor will be explicitly padded as per the provided padding values. Otherwise,
/// the input tensor remains unchanged, but the function will indicate the type of padding
/// (either `Same` or `Valid`) that should be applied during the convolution operation.
///
/// # initializers:
/// * `auto_pad`: The value of the `auto_pad` attribute from the ONNX node.
/// * `attributes`: Extracted attributes from the ONNX node.
/// * `input_nd_array`: The input ndarray.
/// * `batch_size`: The size of the batch dimension from the input ndarray.
///
/// # Returns:
/// * A tuple containing:
///   - `Padding`: The type of padding (`Same` or `Valid`).
///   - `Vec<Array3<f32>>`: A vector of processed inputs, which may be the same as the original
///     inputs or explicitly padded depending on the `auto_pad` value.
///
/// # Panics:
/// * If an unsupported value for `auto_pad` is provided.
///
fn determine_padding_and_input(
    auto_pad: &str,
    attributes: &std::collections::HashMap<String, Attribute<String>>,
    input_nd_array: &ArrayD<f32>,
    batch_size: usize,
) -> (Padding, Vec<Array3<f32>>) {
    match auto_pad {
        "NOT_SET" => {
            let pads = get_ints_attribute(attributes, "pads", Some(vec![0, 0, 0, 0])).unwrap();
            let padded_input: Vec<_> = (0..batch_size)
                .map(|i| {
                    let single_input = input_nd_array.slice(s![i, .., .., ..]).to_owned();
                    pad_matrix_3d(&single_input, &pads).unwrap()
                })
                .collect();
            (Padding::Valid, padded_input)
        }
        _ => {
            let inputs: Vec<_> = (0..batch_size)
                .map(|i| input_nd_array.slice(s![i, .., .., ..]).to_owned())
                .collect();
            (
                match auto_pad {
                    "SAME_UPPER" | "SAME_LOWER" => Padding::Same,
                    "VALID" => Padding::Valid,
                    _ => panic!("Invalid auto_pad value"),
                },
                inputs,
            )
        }
    }
}

/// `conv` - ONNX Node Implementation for Convolution
///
/// Carries out the convolution operation on an input tensor using the provided filter.
/// The operation leverages the `im2col` technique, which reshapes the input tensor into a matrix
/// and then performs matrix multiplication with the filter. This optimizes convolution by
/// reusing computed values, making it more efficient.
///
/// Additionally, the convolution operation is parallelized across batches. If the input has a
/// batch dimension of size N, N distinct convolutions will run concurrently, one for each input
/// in the batch.
///
/// # Arguments
///
/// * `inputs` - A reference to the input tensor containing the data to be convolved.
/// * `initializers` - A reference to the vector containing the filter tensors for the convolution.
/// * `node` - A reference to the ONNX NodeProto containing node-specific data and attributes,
///   such as padding type and strides.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Returns the convoluted output as a `TensorProto` or
///   an error (`OnnxError`) if the operation fails at any stage.
///
/// # Errors
///
/// This function may error if:
/// * The input tensor's shape isn't 4-dimensional.
/// * Attribute extraction fails or provides invalid values.
/// * There's a shape mismatch during the operation.
///
/// # Example
///
/// ```rust
/// let convoluted_output = conv(&input_tensor, &filter_tensors, &node);
/// ```
pub fn conv(
    inputs: &TensorProto,
    initializers: &Vec<&TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Extract the attributes from the node.
    let attributes = extract_attributes(node.get_attribute())?;

    // Convert the input TensorProto to a ndarray.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(inputs)?;
    let batch_size = input_nd_array.shape()[0];

    // Check if the input tensor has the expected 4D shape.
    if input_nd_array.ndim() != 4 {
        return Err(OnnxError::ShapeError("Expected a 4D tensor".to_string()));
    }

    // Get the auto_pad attribute and create the kernel array.
    let auto_pad = get_string_attribute(&attributes, "auto_pad", Some("NOT_SET".to_string()))?;
    let kernel = tensor_proto_to_ndarray::<f32>(initializers[0])?
        .into_shape((
            initializers[0].get_dims()[0] as usize,
            initializers[0].get_dims()[1] as usize,
            initializers[0].get_dims()[2] as usize,
            initializers[0].get_dims()[3] as usize,
        ))
        .map_err(|_| OnnxError::ShapeError("Failed to create kernel matrix".to_string()))?;

    let bias_option = initializers
        .get(1)
        .map(|bias| tensor_proto_to_ndarray::<f32>(bias).unwrap())
        .and_then(|array| array.into_dimensionality::<Ix1>().ok());

    // Determine the padding and optionally pre-pad the input.
    let (padding_mode, input) =
        determine_padding_and_input(&auto_pad, &attributes, &input_nd_array, batch_size);

    let strides = get_ints_attribute(&attributes, "strides", None)?;
    let stride = strides
        .get(0)
        .ok_or(OnnxError::InternalError(
            "Failed to fetch stride attribute".to_string(),
        ))?
        .clone() as usize;

    let group: i64 = get_int_attribute(&attributes, "group", Some(1))?; // default value 1

    let channels = input[0].shape()[0] as i64; // Assuming [channels, height, width]
    let channels_per_group = channels / group;

    if channels % group != 0 {
        return Err(OnnxError::ShapeError(
            "Number of channels is not divisible by group.".to_string(),
        ));
    }

    let kernels_per_group = kernel.shape()[0] as i64 / group;

    // Parallelize the convolution operation for each input in the batch.
    let result_list: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let current_input = &input[i];
            let group_results: Vec<_> = (0..group)
                .into_iter()
                .map(|g| {
                    let group_input = current_input
                        .slice(s![
                            (g * channels_per_group) as usize
                                ..((g + 1) * channels_per_group) as usize,
                            ..,
                            ..
                        ])
                        .to_owned();

                    let group_kernel = kernel
                        .slice(s![
                            (g * kernels_per_group) as usize
                                ..((g + 1) * kernels_per_group) as usize,
                            ..,
                            ..,
                            ..
                        ])
                        .to_owned();

                    let group_bias = if let Some(ref bias) = bias_option {
                        let bias_per_group = bias.shape()[0] as i64 / group;

                        Some(
                            bias.slice(s![
                                (g * bias_per_group) as usize..((g + 1) * bias_per_group) as usize,
                            ])
                            .to_owned(),
                        )
                    } else {
                        None
                    };

                    ConvolutionLayer::new(group_kernel, group_bias, stride, padding_mode)
                        .convolve(&group_input)
                })
                .collect();

            // Convert each result into an ArrayView
            let views: Vec<_> = group_results.iter().map(|arr| arr.view()).collect();

            ndarray::concatenate(Axis(0), &views[..]).unwrap()
        })
        .collect();

    let result = stack_along_batch_dimension(result_list)?;

    // Convert the result to an output tensor and return.
    convert_to_output_tensor(node, result)
}
