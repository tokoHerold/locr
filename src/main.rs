mod core;
mod detector;
mod models;
mod recongizer;

use image::ImageReader;
use ort::session::Session;

use crate::{
    core::device::Device,
    detector::Detector,
    models::{detection::PaddleDetector, recognition::PaddleRecognizer},
    recongizer::Recognizer,
};

include!(concat!(env!("OUT_DIR"), "/dictionary.rs"));

fn main() {
    const DEVICE: Device = Device::Cpu;
    // const DEVICE: Device = Device::Cuda;
    let detector = PaddleDetector::new();
    let mut session = DEVICE
        .configure_session(Session::builder().unwrap())
        .unwrap()
        .commit_from_file("data/models/det.onnx")
        .expect("Failed to load detection model!");
    let image = ImageReader::open("data/test.png")
        .expect("Could not load image")
        .decode()
        .expect("Invalid image format");
    let detection_results = detector.detect(&mut session, &image);
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

    session = DEVICE
        .configure_session(Session::builder().unwrap())
        .unwrap()
        .commit_from_file("data/models/rec.onnx")
        .expect("Failed to load recognition model!");
    let recongizer = PaddleRecognizer {};
    let results = recongizer.recognize(
        &mut session,
        &image.as_rgb8().expect("Wrong image format"),
        bounding_boxes,
        &CHARACTER_DICT,
    );

    for result in results {
        println!("{:}", result.text);
    }

    // println!("{:?}", results);
    println!("Hello, world!");
}
