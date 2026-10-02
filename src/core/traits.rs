use image::{DynamicImage, RgbImage};

use crate::core::{
    error::OcrError,
    types::{BoundingBox, DetectionResult, OcrResult, RecognitionResult},
};

/// Text detection contract responsible for localizing text regions in an image.
pub trait TextDetector {
    /// Retrieves bounding boxes for all text occurences in a supplied image.
    ///
    /// # Arguments
    ///
    /// * `session` - Handle to ONNX session
    /// * `image` - Image to detect text on, in RGB format
    ///
    /// # Returns
    /// A list of bounding boxes for each text element.
    ///
    fn detect(&mut self, image: &RgbImage) -> Result<Vec<DetectionResult>, OcrError>;
}

/// Text recognition contract responsible for transcribing cropped text regions.
pub trait TextRecognizer {
    /// Recognizes the text on an image inside supplied bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `session` - Handle to ONNX session
    /// * `image` - Image in RGB format
    /// * `bounding_boxes` - List of bounding boxes on the image, containing text occurences
    ///
    /// # Returns
    ///
    /// Extracted text for all bounding boxes, with their confidence score.
    fn recognize(
        &mut self,
        image: &RgbImage,
        bounding_boxes: Vec<&BoundingBox>,
    ) -> Result<Vec<RecognitionResult>, OcrError>;
}

/// High-level OCR engine contract for end-to-end text processing.
///
/// Implemented by classical two-stage pipelines as well as end-to-end vision-language models.
pub trait OcrEngine {
    /// Processes an input image and returns detected text items and run statistics.
    ///
    /// # Errors
    ///
    /// Returns [`OcrError`] if detection, recognition, or image preprocessing fails.
    fn process(&mut self, image: &DynamicImage) -> Result<OcrResult, OcrError>;
}
