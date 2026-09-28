mod detector;
mod models;
mod recongizer;

use image::ImageReader;
use ort::session::Session;

use crate::{detector::Detector, models::detection::PaddleDetector};

fn main() {
    let detector = PaddleDetector::new();
    let mut session = Session::builder()
        .unwrap()
        .commit_from_file("data/models/det.onnx")
        .expect("Failed to load detection model!");
    let image = ImageReader::open("data/test.png")
        .expect("Could not load image")
        .decode()
        .expect("Invalid image format");
    let bounding_boxes = detector.detect(&mut session, &image);
    for bb in bounding_boxes {
        println!(
            "Detected text at pixel ({:}, {:}), width: {:}, height: {:}",
            bb.x, bb.y, bb.width, bb.height
        )
    }

    println!("Hello, world!");
}
