use image::DynamicImage;
use ndarray::Array4;
use ort::session::SessionOutputs;

use crate::detector::BoundingBox;

pub struct RecognitionResult {
    text: String,
    score: f32,
}

pub trait Recognizer {
    /// Processes an image for model inference.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format
    ///
    /// # Returns
    ///
    /// A tensor that can be inserted into the input layer of the model.
    fn preprocess(&self, image: &DynamicImage) -> Array4<f32>;

    /// Extracts decoded text and confidence from the model output
    ///
    /// # Arguments
    ///
    /// * `model_output` - Output layer from the model
    ///
    /// # Returns
    ///
    /// A list of bounding boxes around each text segment.
    fn decoode(&self, model_output: &SessionOutputs) -> RecognitionResult;

    /// Recognizes the text on an image inside supplied bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `image` - Image in RGB format
    /// * `bounding_boxes` - List of bounding boxes on the image, containing text occurences
    ///
    /// # Returns
    ///
    /// Extracted text for all bounding boxes, with their confidence score.
    fn recognize(&self, image: &DynamicImage, bounding_boxes: Vec<BoundingBox>) -> Vec<RecognitionResult>;
}


