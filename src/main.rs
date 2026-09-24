mod models;
mod detector;
mod recongizer;

use crate::models::detection::PaddleDetector;

fn main() {
    let _detector  = PaddleDetector::new("models/detection.onnx");
    println!("Hello, world!");
}
