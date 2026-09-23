use image::DynamicImage;
use ndarray::Array4;
use ort::session::SessionOutputs;

use crate::detector::{BoundingBox, Detector};

struct PaddleDetector {
    model_path: String,
    detection_threshold: f32,
    box_threshold: f32,
    unclip_ratio: f32,
    limit_side_length: u32,
}

impl PaddleDetector {
    pub fn new(model_path: &str) -> Self {
        Self {
            model_path: model_path.to_string(),
            detection_threshold: 0.3,
            box_threshold: 0.5,
            unclip_ratio: 1.6,
            limit_side_length: 960,
        }
    }
}

impl Detector for PaddleDetector {
    fn preprocess(&self, image: &DynamicImage) -> Array4<f32> {
        todo!()
    }

    fn postprocess(&self, model_output: &SessionOutputs) -> Vec<BoundingBox> {
        todo!()
    }

    fn infer(&self, input: &Array4<f32>) -> SessionOutputs<'_> {
        todo!()
    }
}
