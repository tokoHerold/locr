use image::DynamicImage;
use ndarray::Array4;
use ort::session::SessionOutputs;

use crate::recongizer::{RecognitionResult, Recognizer};

struct PaddleRecognizer {
}

impl Recognizer for PaddleRecognizer {
    fn preprocess(&self, crop: &image::SubImage<&DynamicImage>) -> Array4<f32> {
        todo!()
    }

    fn decoode(&self, model_output: &SessionOutputs) -> RecognitionResult {
        todo!()
    }

    fn infer<'a>(&self, session: &'a mut ort::session::Session, input: &Array4<f32>) -> SessionOutputs<'a> {
        todo!()
    }
}
