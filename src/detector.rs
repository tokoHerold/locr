use image::DynamicImage;
use ndarray::Array4;
use ort::session::SessionOutputs;

pub struct BoundingBox {
    x: u32,
    y: u32,
    width: u32,
    height: u32
}

pub trait Detector {
    /// Processes an image for model inference.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format
    ///
    /// # Returns
    /// A tensor that can be inserted into the input layer of the model.
    ///
    fn preprocess(&self, image: &DynamicImage) -> Array4<f32>;

    /// Extracts bounding boxes from the model output format
    ///
    /// # Arguments
    ///
    /// * `model_output` - Output layer from the model
    ///
    /// # Returns
    ///
    /// A list of bounding boxes around each text segment.
    fn postprocess(&self, model_output: &SessionOutputs) -> Vec<BoundingBox>;

    /// Runs the data through the underlying model
    ///
    /// # Arguments
    ///
    /// * `input` - Model input tensor
    ///
    /// # Returns
    ///
    /// Ouput of the model
    fn infer(&self, input: &Array4<f32>) -> SessionOutputs<'_>;

    /// Retrieves bounding boxes for all text occurences in a supplied image.
    ///
    /// # Arguments
    ///
    /// * `image` - Image to detect text on, in RGB format
    ///
    /// # Returns
    /// A list of bounding boxes for each text element.
    ///
    fn detect(&self, image: &DynamicImage) -> Vec<BoundingBox> {
        let tensor = self.preprocess(image);
        let model_output = self.infer(&tensor);
        self.postprocess(&model_output)
    }
}


