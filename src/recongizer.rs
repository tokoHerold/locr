use image::{DynamicImage, SubImage, imageops::crop_imm};
use ndarray::Array4;
use ort::session::{Session, SessionOutputs};

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
    /// * `image` - RGB crop to run recognition on
    ///
    /// # Returns
    ///
    /// A tensor that can be inserted into the input layer of the model.
    fn preprocess(&self, crop: &SubImage<&DynamicImage>) -> Array4<f32>;

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

    /// Runs the data through the underlying model
    ///
    /// # Arguments
    ///
    /// * `input` - Model input tensor
    ///
    /// # Returns
    ///
    /// Ouput of the model
    fn infer<'a>(&self, session: &'a mut Session, input: &Array4<f32>) -> SessionOutputs<'a>;

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
    fn recognize(&self, session: &mut Session, image: &DynamicImage, bounding_boxes: Vec<BoundingBox>) -> Vec<RecognitionResult> {
        bounding_boxes.iter().map(|bb| -> RecognitionResult {
            let crop = crop_imm(image, bb.x, bb.y, bb.width, bb.height);
            let tensor = self.preprocess(&crop);
            let model_output = self.infer(session, &tensor);
            self.decoode(&model_output)
        }).collect()
    }
}


