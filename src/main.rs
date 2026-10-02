mod core;
mod models;
mod pipeline;

use image::ImageReader;

use crate::{
    core::{
        device::Device,
        traits::{TextDetector, TextRecognizer},
    },
    models::{ppv6_detection::PaddleDetector, ppv6_recognition::PaddleRecognizer},
};

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

    let mut recongizer = PaddleRecognizer::new(DEVICE).expect("Failed to create recognizer");
    let results = recongizer.recognize(&image, bounding_boxes);

    for result in results {
        println!("{:}", result.text);
    }

    // println!("{:?}", results);
    println!("Hello, world!");
}
