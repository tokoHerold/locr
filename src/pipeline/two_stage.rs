use std::time::Instant;

use image::DynamicImage::ImageRgb8;

use crate::core::{
    error::OcrError,
    traits::{OcrEngine, TextDetector, TextRecognizer},
    types::{DetectionResult, OcrResult},
};

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
        Self {
            detector,
            recognizer,
        }
    }
}

impl<D: TextDetector, R: TextRecognizer> OcrEngine for TwoStagePipeline<D, R> {
    /// Extracts text from a supplied image by first detecting possible text regions, placing
    /// bounding boxes around those regions, and running a character recognition on each.
    ///
    /// This should work for most printed and some handwritten text, with the following limitations:
    /// - Text lines must be horizontal
    /// - Lines of text must not overlap
    /// - For high resolution images, small text might not be detected
    ///
    /// # Arguments
    ///
    /// * `image` - Image (ideally in RGB format) to run OCR on
    ///
    /// # Returns
    /// An [`OcrResult`] containing all bounding boxes with their extracted text, confidence scores
    /// and execution walltime.
    ///
    /// # Errors
    /// Returns an [`OcrError`] if text detection or recognition fails.
    ///
    /// # Examples
    /// ```rust
    /// use locr::{default_engine, Device, OcrEngine};
    /// use image::open;
    ///
    /// let mut engine = default_engine(Device::Cpu).unwrap();
    /// let img = open("document.png").unwrap();
    /// let result = engine.process(&img).unwrap();
    /// println!("Found {} text blocks in {} ms", result.items.len(), result.processing_time_ms);
    /// ```
    fn process(&mut self, image: &image::DynamicImage) -> Result<OcrResult, OcrError> {
        let start = Instant::now();
        let rgb_image = if let ImageRgb8(image) = image {
            image
        } else {
            &image.to_rgb8() // Create copy if image is in wrong format
        };
        let detection_results = self.detector.detect(&rgb_image)?;
        let recognition_results = self.recognizer.recognize(&rgb_image, &detection_results)?;
        Ok(OcrResult {
            items: detection_results.iter().cloned().zip(recognition_results).collect(),
            processing_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}
