extern crate protoc_rust;
use std::env;
use std::process::Command;

#[allow(unused_macros)]
macro_rules! src_dir {
    () => {
        "third_party/onnx"
    };
    ($src_path:expr) => {
        concat!(src_dir!(), "/", $src_path)
    };
}

fn protoc_installed() -> bool {
    Command::new("protoc").arg("--version").output().is_ok()
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    if protoc_installed() {
        protoc_rust::Codegen::new()
            .out_dir("src/onnx_rustime/onnx_proto")
            .inputs(&[
                src_dir!("onnx-ml.proto3"),
                src_dir!("onnx-operators-ml.proto3"),
                src_dir!("onnx-data.proto3"),
            ])
            .include(src_dir!())
            .run()
            .expect("Failed to run protoc");
    } else {
        println!("cargo:warning=protoc not found. Skipping code generation.");
    }
}
