mod core;
mod models;

use std::path::PathBuf;

use image::ImageReader;
use ort::{compiler::ModelCompiler, session::Session};

use crate::{
    core::{device::Device, traits::{TextDetector, TextRecognizer}}, models::{ppv6_detection::PaddleDetector, recognition::PaddleRecognizer},
};

include!(concat!(env!("OUT_DIR"), "/dictionary.rs"));

fn main() {
    // const DEVICE: Device = Device::Cpu;
    const DEVICE: Device = Device::Cuda;
    let mut detector = PaddleDetector::new(DEVICE).expect("Failed to create detector");
    let image = ImageReader::open("data/test.png")
        .expect("Could not load image")
        .decode()
        .expect("Invalid image format");
    let image = image.as_rgb8().expect("Expected RGB image");
    let detection_results = detector.detect(&image);
    for result in &detection_results {
        let bb = &result.bounding_box;
        println!(
            "Detected text at pixel ({:}, {:}), width: {:}, height: {:} - confidence {}",
            bb.x, bb.y, bb.width, bb.height, result.score
        )
    }

    let bounding_boxes = detection_results
        .iter()
        .map(|r| r.bounding_box.clone())
        .collect();

    let mut session_options = DEVICE
        .configure_session(Session::builder().unwrap()).unwrap();
        // .commit_from_file("data/models/rec.onnx")
        // .expect("Failed to load recognition model!");
    let compiled_path = PathBuf::from("data/models/compiled.onnx");
    if !compiled_path.exists() {

    ModelCompiler::new(session_options.clone()).unwrap().with_model_from_file("data/models/PP_OCRv6_tiny_rec.onnx")
        .unwrap().compile_to_file(&compiled_path).expect("Failed to compile rec model");
    }
    let mut session = session_options.commit_from_file(&compiled_path).expect("Failed to load compiled model");
    let recongizer = PaddleRecognizer {};
    let results = recongizer.recognize(
        &mut session,
        &image,
        bounding_boxes,
        &CHARACTER_DICT,
    );

    for result in results {
        println!("{:}", result.text);
    }

    // println!("{:?}", results);
    println!("Hello, world!");
}
