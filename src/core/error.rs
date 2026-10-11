use std::process::ExitCode;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[cfg(feature = "backend-ort")]
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[cfg(feature = "backend-ort")]
    #[error("Device execution provider error: {0}")]
    DeviceUnavailable(#[from] ort::Error<ort::session::builder::SessionBuilder>),

    #[error("Invalid model output: {0}")]
    ModelOutputError(String),

    #[error("Feature is not active: {0}")]
    FeatureDisabled(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Dimension or shape mismatch: {0}")]
    DimensionMismatch(#[from] ndarray::ShapeError),
}

impl OcrError {
    pub fn report(self) -> std::process::ExitCode {
        eprintln!("{self}");
        ExitCode::FAILURE
    }
}
