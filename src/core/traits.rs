use image::RgbImage;

use crate::core::types::{BoundingBox, DetectionResult, RecognitionResult};

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
    fn detect(&mut self, image: &RgbImage) -> Vec<DetectionResult>;
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
        bounding_boxes: Vec<BoundingBox>,
    ) -> Vec<RecognitionResult>;
}
