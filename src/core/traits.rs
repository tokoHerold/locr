use image::RgbImage;
use ndarray::Array4;
use ort::session::{Session, SessionOutputs};

use crate::core::types::{BoundingBox, DetectionResult, RecognitionResult};

pub trait TextDetector {
    /// Processes an image for model inference.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format
    ///
    /// # Returns
    /// A tensor that can be inserted into the input layer of the model.
    ///
    fn preprocess(&self, image: &RgbImage) -> Array4<f32>;

    /// Extracts bounding boxes from the model output format
    ///
    /// # Arguments
    ///
    /// * `model_output` - Output layer from the model
    ///
    /// # Returns
    ///
    /// A list of bounding boxes around each text segment.
    fn postprocess(
        &self,
        model_output: &SessionOutputs,
        image: &RgbImage,
    ) -> Vec<DetectionResult>;

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
    fn detect(&self, session: &mut Session, image: &RgbImage) -> Vec<DetectionResult> {
        let tensor = self.preprocess(image);
        let model_output = self.infer(session, &tensor);
        self.postprocess(&model_output, image)
    }
}

pub trait TextRecognizer {
    /// Processes an image for model inference.
    ///
    /// # Arguments
    ///
    /// * `image` - reference to RGB image
    /// * `bounding_box` - box to crop image
    ///
    /// # Returns
    ///
    /// A tensor that can be inserted into the input layer of the model.
    fn preprocess(&self, image: &RgbImage, bounding_box: &BoundingBox) -> Array4<f32>;

    /// Extracts decoded text and confidence from the model output
    ///
    /// # Arguments
    ///
    /// * `model_output` - Output layer from the model
    ///
    /// # Returns
    ///
    /// A list of bounding boxes around each text segment.
    fn decoode(&self, model_output: &SessionOutputs, dict: &[char]) -> RecognitionResult;

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
    fn recognize(
        &self,
        session: &mut Session,
        image: &RgbImage,
        bounding_boxes: Vec<BoundingBox>,
        dict: &[char],
    ) -> Vec<RecognitionResult> {
        bounding_boxes
            .iter()
            .map(|bounding_box| -> RecognitionResult {
                let tensor = self.preprocess(image, bounding_box);
                let model_output = self.infer(session, &tensor);
                self.decoode(&model_output, dict)
            })
            .collect()
    }
}
