use image::DynamicImage;
use ndarray::{Array4, ArrayViewD};
use ort::session::Session;

pub trait OcrRecognizer {

    /// Transforms an image into a Tensor (RGBA), required by the model.
    ///
    /// # Arguments
    ///
    /// * `img` - Image to perform OCR on
    ///
    /// # Returns
    /// A specific 4D-Tensor, which can be directly used as an input to the specific OCR model
    ///
    /// # Examples
    /// ```rust
    /// let dynamic_img = DynamicImage::ImageRgba8(img_buffer);
    /// let input_tensor = model.preprocess(&dynamic_img);
    /// ```
    fn preprocess(&self, img: &DynamicImage) -> Array4<f32>;

    /// Transforms an output tensor, retrieved from a model session, into text.
    ///
    /// # Arguments
    ///
    /// * `output_tensor` - Model result tensor
    ///
    fn decode(&self, output_tensor: ArrayViewD<f32>) -> String;

    /// Retrieves a reference to the loaded [ONNX](https://onnxruntime.ai/) session
    ///
    /// # Examples
    /// ```rust
    /// let session = model.session();
    /// let outputs = session.run(inputs![model.input_name() => input_tensor.view()].unwrap())
    /// .expect("Error evaluating model");
    /// ```
    fn session(&self) -> &Session;

    /// Retrieves the name of the input layer (often 'x', 'input' or 'image', depends on the model).
    fn input_name(&self) -> &str;
}
