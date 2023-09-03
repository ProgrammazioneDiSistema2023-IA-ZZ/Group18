use crate::onnx_rustime::onnx_proto::onnx_ml_proto3::*;
use crate::onnx_rustime::backend::helper::OnnxError;
use protobuf::{CodedInputStream, Message};

pub fn parse_raw_data_as_floats(raw_data: &[u8]) -> Vec<f32> {
    let mut doubles = Vec::with_capacity(raw_data.len() / 4);

    for i in (0..raw_data.len()).step_by(4) {
        let bytes = [
            raw_data[i],
            raw_data[i + 1],
            raw_data[i + 2],
            raw_data[i + 3],
        ];
        let double_value = f32::from_le_bytes(bytes);
        doubles.push(double_value);
    }

    doubles
}

pub fn parse_raw_data_as_ints64(raw_data: &[u8]) -> Vec<i64> {
    let mut ints64 = Vec::with_capacity(raw_data.len() / 8);

    for i in (0..raw_data.len()).step_by(8) {
        let bytes = [
            raw_data[i],
            raw_data[i + 1],
            raw_data[i + 2],
            raw_data[i + 3],
            raw_data[i + 4],
            raw_data[i + 5],
            raw_data[i + 6],
            raw_data[i + 7],
        ];
        let int64_value = i64::from_le_bytes(bytes);
        ints64.push(int64_value);
    }

    ints64
}

pub struct OnnxParser;

impl OnnxParser {
    pub fn load_model(path: String) -> Result<ModelProto, OnnxError> {
        let mut file = std::fs::File::open(path).map_err(|_| OnnxError::InternalError("Failed to open model file".to_string()))?;
        let mut stream = CodedInputStream::new(&mut file);

        let mut model = ModelProto::new();
        model.merge_from(&mut stream).map_err(|_| OnnxError::InternalError("Failed to merge model from stream".to_string()))?;

        Ok(model)
    }

    pub fn load_data(path: String) -> Result<TensorProto, OnnxError> {
        let mut file = std::fs::File::open(path).map_err(|_| OnnxError::InternalError("Failed to open data file".to_string()))?;
        let mut stream = CodedInputStream::new(&mut file);

        let mut tensor = TensorProto::new();
        tensor.merge_from(&mut stream).map_err(|_| OnnxError::InternalError("Failed to merge tensor from stream".to_string()))?;

        Ok(tensor)
    }

    #[allow(dead_code)]
    pub fn save_model(model: &ModelProto, path: String) -> Result<(), OnnxError> {
        let mut file = std::fs::File::create(path).map_err(|_| OnnxError::InternalError("Failed to create model file".to_string()))?;
        model.write_to_writer(&mut file).map_err(|_| OnnxError::InternalError("Failed to write model to file".to_string()))?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn save_data(tensor: &TensorProto, path: String) -> Result<(), OnnxError> {
        let mut file = std::fs::File::create(path).map_err(|_| OnnxError::InternalError("Failed to create data file".to_string()))?;
        tensor.write_to_writer(&mut file).map_err(|_| OnnxError::InternalError("Failed to write tensor to file".to_string()))?;
        Ok(())
    }
}
