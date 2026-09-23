use image::DynamicImage;
use ndarray::Array4;
use ort::session::SessionOutputs;

use crate::{detector::BoundingBox, recongizer::{RecognitionResult, Recognizer}};

struct PaddleRecognizer {
    model_path: String,
    config_path: String,
}

impl PaddleRecognizer {
    pub fn new(model_path: &str, config_path: &str) -> Self {
        Self {
            model_path: model_path.to_string(),
            config_path: config_path.to_string(),
        }
    }
}

impl Recognizer for PaddleRecognizer {
    fn preprocess(&self, image: &DynamicImage) -> Array4<f32> {
        todo!()
    }

    fn decoode(&self, model_output: &SessionOutputs) -> RecognitionResult {
        todo!()
    }

    fn recognize(&self, image: &DynamicImage, bounding_boxes: Vec<BoundingBox>) -> Vec<RecognitionResult> {
        todo!()
    }
}
