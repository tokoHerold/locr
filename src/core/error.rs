use ort::session::builder::SessionBuilder;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Device execution provider error: {0}")]
    DeviceUnavailable(#[from] ort::Error<SessionBuilder>),

    #[error("Invalid model output: {0}")]
    ModelOutputError(String),

    #[error("Feature is not active: {0}")]
    FeatureDisabled(String),
}
