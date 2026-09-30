mod detector;
mod models;
mod recongizer;


use image::ImageReader;
use ort::session::Session;
use serde::Deserialize;

use crate::{
    detector::Detector, models::{detection::PaddleDetector, recognition::PaddleRecognizer}, recongizer::Recognizer,
};

#[derive(Deserialize, Debug)]
struct Config {
    #[serde(rename = "PostProcess")]
    post_process: PostProcessConfig,
}

#[derive(Deserialize, Debug)]
struct PostProcessConfig {
    character_dict: Vec<String>,
}

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
    for bb in &bounding_boxes {
        println!(
            "Detected text at pixel ({:}, {:}), width: {:}, height: {:}",
            bb.x, bb.y, bb.width, bb.height
        )
    }

    const YAML_CONTENT: &'static str = include_str!("../data/models/inference.yml");
    let dict = serde_saphyr::from_str::<Config>(YAML_CONTENT).expect("Failed to parse YAML");
    let mut characters: Vec<String> =
        Vec::with_capacity(dict.post_process.character_dict.len() + 2);

    characters.push("<blank>".to_string());
    characters.extend(dict.post_process.character_dict);
    characters.push(" ".to_string());

    session = Session::builder()
        .unwrap()
        .commit_from_file("data/models/rec.onnx")
        .expect("Failed to load recognition model!");
    let recongizer = PaddleRecognizer {};
    let results = recongizer.recognize(
        &mut session,
        &image.as_rgb8().expect("Wrong image format"),
        bounding_boxes,
        &characters,
    );

    for result in results {
        println!("{:}", result.text);
    }

    // println!("{:?}", results);
    println!("Hello, world!");
}
