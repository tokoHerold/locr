mod core;
mod models;
mod pipeline;

use image::ImageReader;

use crate::{
    core::{device::Device, error::OcrError, traits::OcrEngine},
    models::{ppv6_detection::PaddleDetector, ppv6_recognition::PaddleRecognizer},
    pipeline::two_stage::TwoStagePipeline,
};

type PaddleOcr = TwoStagePipeline<PaddleDetector, PaddleRecognizer>;
fn default_engine(device: Device) -> Result<PaddleOcr, OcrError> {
    let detector = PaddleDetector::new(device)?;
    let recognizer = PaddleRecognizer::new(device)?;
    Ok(PaddleOcr::new(detector, recognizer))
}

fn main() {
    // const DEVICE: Device = Device::Cpu;
    const DEVICE: Device = Device::Cuda { device_id: 0 };
    let image = ImageReader::open("data/test.png")
        .expect("Could not load image")
        .decode()
        .expect("Invalid image format");

    let mut engine = default_engine(DEVICE).unwrap();
    let result = engine.process(&image).unwrap();
    println!(
        "Found {} text blocks in {} ms",
        result.items.len(),
        result.processing_time_ms
    );
    println!("{:?}", result.items);

    // println!("{:?}", results);
    println!("Hello, world!");
}
