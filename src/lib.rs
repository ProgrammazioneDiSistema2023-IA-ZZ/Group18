pub mod display;
pub mod onnx_rustime;

pub use onnx_rustime::backend;
pub use onnx_rustime::onnx_proto;
pub use onnx_rustime::ops;

#[cfg(any(feature = "include_pyo3", feature = "include_neon"))]
mod common {
    use super::*;

    use once_cell::sync::Lazy;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub use display::display_outputs;

    pub use onnx_rustime::backend::parser::OnnxParser;
    pub use onnx_rustime::backend::run::run;
    pub use onnx_rustime::onnx_proto::onnx_ml_proto3::ModelProto;
    pub use onnx_rustime::onnx_proto::onnx_ml_proto3::TensorProto;
    pub use onnx_rustime::ops::utils::tensor_proto_to_ndarray;

    pub use onnx_rustime::shared::Model;
    pub use onnx_rustime::shared::MODEL_NAME;
    pub use onnx_rustime::shared::VERBOSE;

    pub type ModelId = usize; // An alias for our model IDs.
    pub type DataId = usize;

    pub static MODELS: Lazy<Mutex<HashMap<ModelId, ModelProto>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));
    pub static DATA: Lazy<Mutex<HashMap<DataId, TensorProto>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));

    pub fn store_model(model: ModelProto) -> ModelId {
        let mut models = MODELS.lock().unwrap();
        let id = models.len() + 1; // Generate a new unique ID.
        models.insert(id, model);
        id
    }

    pub fn get_model(id: ModelId) -> Option<ModelProto> {
        let models = MODELS.lock().unwrap();
        models.get(&id).cloned() // Cloned because we can't return a reference directly.
    }

    pub fn store_data(tensor: TensorProto) -> DataId {
        let mut data = DATA.lock().unwrap();
        let id = data.len() + 1; // Generate a new unique ID.
        data.insert(id, tensor);
        id
    }

    pub fn get_data(id: DataId) -> Option<TensorProto> {
        let data = DATA.lock().unwrap();
        data.get(&id).cloned() // Cloned because we can't return a reference directly.
    }
}

#[cfg(feature = "include_pyo3")]
mod include_pyo3 {
    use super::common::*;

    use pyo3::prelude::*;
    use pyo3::wrap_pyfunction;

    #[pyfunction]
    pub fn py_load_model(path: &str) -> PyResult<ModelId> {
        let model_enum = match path {
            "models/bvlcalexnet-12/bvlcalexnet-12.onnx" => Model::AlexNet,
            "models/caffenet-12/caffenet-12.onnx" => Model::CaffeNet,
            "models/mnist-8/mnist-8.onnx" => Model::Mnist,
            "models/resnet152-v2-7/resnet152-v2-7.onnx" => Model::ResNet,
            "models/squeezenet1.0-12/squeezenet1.0-12.onnx" => Model::SqueezeNet,
            "models/zfnet512-12/zfnet512-12.onnx" => Model::ZFNet,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyException, _>(
                    "Invalid model path",
                ))
            }
        };

        {
            let mut d = MODEL_NAME.lock().unwrap();
            *d = model_enum;
        }

        match OnnxParser::load_model(path.to_string()) {
            Ok(model) => {
                let id = store_model(model);
                Ok(id)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                "{:?}",
                e
            ))),
        }
    }

    #[pyfunction]
    pub fn py_load_data(path: &str) -> PyResult<DataId> {
        match OnnxParser::load_data(path.to_string()) {
            Ok(tensor) => {
                let id = store_data(tensor);
                Ok(id)
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                "{:?}",
                e
            ))),
        }
    }

    #[pyfunction]
    pub fn py_print_data(data_id: DataId) -> PyResult<()> {
        // Get the TensorProto from the DATA storage using the provided ID
        if let Some(tensor) = get_data(data_id) {
            // Convert the TensorProto to an ndarray
            let ndarray = tensor_proto_to_ndarray::<f32>(&tensor)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{:?}", e)))?;

            // Print the ndarray
            println!("{:?}", ndarray);

            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid data ID",
            ))
        }
    }

    #[pyfunction]
    pub fn py_run(model_id: ModelId, input_data_id: DataId, verbose: bool) -> PyResult<DataId> {
        // Retrieve the ModelProto and TensorProto from storages
        let model = get_model(model_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid model ID"))?;

        let input_tensor = get_data(input_data_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid data ID"))?;

        {
            let mut v = VERBOSE.lock().unwrap();
            *v = verbose;
        }

        // Call the original run function
        let output_tensor = run(&model, input_tensor);

        // Store the output TensorProto in the DATA storage and return its ID
        Ok(store_data(output_tensor))
    }

    #[pyfunction]
    pub fn py_display_outputs(
        predicted_data_id: DataId,
        expected_data_id: Option<DataId>,
    ) -> PyResult<()> {
        // Retrieve the predicted TensorProto from storages
        let predicted_tensor = get_data(predicted_data_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid predicted data ID")
        })?;

        // Retrieve the expected TensorProto from storages, if available
        let expected_tensor = match expected_data_id {
            Some(id) => Some(get_data(id).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid expected data ID")
            })?),
            None => None,
        };

        display_outputs(&predicted_tensor, expected_tensor);

        Ok(())
    }

    #[pymodule]
    fn onnx_rustime_lib(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_load_model, m)?)?;
        m.add_function(wrap_pyfunction!(py_load_data, m)?)?;
        m.add_function(wrap_pyfunction!(py_print_data, m)?)?;
        m.add_function(wrap_pyfunction!(py_run, m)?)?;
        m.add_function(wrap_pyfunction!(py_display_outputs, m)?)?;

        Ok(())
    }
}

#[cfg(feature = "include_neon")]
mod include_neon {
    use super::common::*;
    use neon::prelude::*;

    fn js_load_model(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let path = cx.argument::<JsString>(0)?.value(&mut cx);

        println!("path: {:?}", path);

        let model_enum = match path.as_str() {
            "models/bvlcalexnet-12/bvlcalexnet-12.onnx" => Model::AlexNet,
            "models/caffenet-12/caffenet-12.onnx" => Model::CaffeNet,
            "models/mnist-8/mnist-8.onnx" => Model::Mnist,
            "models/resnet152-v2-7/resnet152-v2-7.onnx" => Model::ResNet,
            "models/squeezenet1.0-12/squeezenet1.0-12.onnx" => Model::SqueezeNet,
            "models/zfnet512-12/zfnet512-12.onnx" => Model::ZFNet,
            _ => return cx.throw_error("Invalid model path"),
        };

        {
            let mut d = MODEL_NAME.lock().unwrap();
            *d = model_enum;
        }

        match OnnxParser::load_model(path) {
            Ok(model) => {
                let id = store_model(model);
                Ok(cx.number(id as f64))
            }
            Err(e) => {
                let err_msg = format!("{:?}", e);
                cx.throw_error(err_msg)
            }
        }
    }

    fn js_load_data(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let path = cx.argument::<JsString>(0)?.value(&mut cx);

        match OnnxParser::load_data(path) {
            Ok(tensor) => {
                let id = store_data(tensor);
                Ok(cx.number(id as f64))
            }
            Err(e) => {
                let err_msg = format!("{:?}", e);
                cx.throw_error(err_msg)
            }
        }
    }

    fn js_print_data(mut cx: FunctionContext) -> JsResult<JsUndefined> {
        let data_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as DataId;

        if let Some(tensor) = get_data(data_id) {
            let ndarray = match tensor_proto_to_ndarray::<f32>(&tensor) {
                Ok(ndarray) => ndarray,
                Err(e) => {
                    let err_msg = format!("{:?}", e);
                    return cx.throw_error(err_msg);
                }
            };

            println!("{:?}", ndarray);

            Ok(cx.undefined())
        } else {
            let err_msg = "Invalid data ID".to_string();
            cx.throw_error(err_msg)
        }
    }

    fn js_run(mut cx: FunctionContext) -> JsResult<JsNumber> {
        let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as ModelId;
        let input_data_id = cx.argument::<JsNumber>(1)?.value(&mut cx) as DataId;
        let verbose = cx.argument::<JsBoolean>(2)?.value(&mut cx);

        let model = match get_model(model_id) {
            Some(model) => model,
            None => {
                let err_msg = "Invalid model ID".to_string();
                return cx.throw_error(err_msg);
            }
        };

        let input_tensor = match get_data(input_data_id) {
            Some(tensor) => tensor,
            None => {
                let err_msg = "Invalid data ID".to_string();
                return cx.throw_error(err_msg);
            }
        };

        {
            let mut v = VERBOSE.lock().unwrap();
            *v = verbose;
        }

        let output_tensor = run(&model, input_tensor);
        let data_id = store_data(output_tensor);

        Ok(cx.number(data_id as f64))
    }

    fn js_display_outputs(mut cx: FunctionContext) -> JsResult<JsUndefined> {
        let predicted_data_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as DataId;
        let expected_data_id = cx.argument::<JsNumber>(1)?.value(&mut cx) as DataId;

        let predicted_tensor = match get_data(predicted_data_id) {
            Some(tensor) => tensor,
            None => {
                let err_msg = "Invalid predicted data ID".to_string();
                return cx.throw_error(err_msg);
            }
        };

        let expected_tensor = match get_data(expected_data_id) {
            Some(tensor) => tensor,
            None => {
                let err_msg = "Invalid expected data ID".to_string();
                return cx.throw_error(err_msg);
            }
        };

        display_outputs(&predicted_tensor, Some(expected_tensor));

        Ok(cx.undefined())
    }

    #[neon::main]
    fn main(mut cx: ModuleContext) -> NeonResult<()> {
        cx.export_function("js_load_model", js_load_model)?;
        cx.export_function("js_load_data", js_load_data)?;
        cx.export_function("js_print_data", js_print_data)?;
        cx.export_function("js_run", js_run)?;
        cx.export_function("js_display_outputs", js_display_outputs)?;

        Ok(())
    }
}
