use image::{DynamicImage, RgbImage};
use ndarray::{ArrayView, Dimension, IxDyn};

use crate::core::{
    error::OcrError,
    types::{DetectionResult, OcrResult, RecognitionResult},
};

/// Generic, shape-agnostic model execution trait.
///
/// Parametrized over input dimension `D: Dimension`.
pub trait Model<D: Dimension> {
    /// Output buffer tied to the model session borrow lifetime `'a`
    type Output<'a>: ModelOutput
    where
        Self: 'a;

    /// Runs forward inference on an [`ndarray`] tensor view of rank `D`.
    ///
    /// # Arguments:
    /// `input` - input layer of the model
    ///
    /// # Returns
    /// A [`ModelOutput`] struct containing the inference results on success, or an [`OcrError`].
    fn forward<'a>(
        &'a mut self,
        input: ArrayView<'a, f32, D>,
    ) -> Result<Self::Output<'a>, OcrError>;
}

/// Output buffer abstraction providing shape inspection and slice access.
pub trait ModelOutput {
    /// Retrieves the original shape of the output tensor
    fn shape(&self) -> &[usize];

    /// Retrieves the model output as a flat slice
    fn raw_data(&self) -> &[f32];

    /// Retrieves an [`ArrayView`] of a specific rank from the model output.
    ///
    /// # Returns
    /// Returns `Ok(ArrayView)` on success, or an error on a dimensionality mismatch.
    fn view<'a, D: Dimension>(&'a self) -> Result<ArrayView<'a, f32, D>, OcrError> {
        Ok(ArrayView::from_shape(IxDyn(self.shape()), self.raw_data())?
            .into_dimensionality::<D>()?)
    }
}

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
        bounding_boxes: &Vec<DetectionResult>,
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
