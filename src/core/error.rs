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

    #[error("Invalid input or dimensions: {0}")]
    InvalidInput(String),
}
