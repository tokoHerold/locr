use crate::core::traits::{OcrEngine, TextDetector, TextRecognizer};

/// End-to-end OCR pipeline orchestrating a decoupled detector and recognizer.
pub struct TwoStagePipeline<D: TextDetector, R: TextRecognizer> {
    /// Text detection backend
    pub detector: D,
    /// Text recognition backend
    pub recognizer: R,
}

impl<D: TextDetector, R: TextRecognizer> TwoStagePipeline<D, R> {
    /// Creates a new pipeline from a detector and a recognizer.
    pub fn new(detector: D, recognizer: R) -> Self {
        Self { detector, recognizer }
    }
}

impl<D: TextDetector, R: TextRecognizer> OcrEngine for TwoStagePipeline<D, R> {
    fn process(&mut self, image: &image::DynamicImage) -> Result<crate::core::types::OcrResult, crate::core::error::OcrError> {
        todo!()
    }
}
