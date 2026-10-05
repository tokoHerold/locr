use std::process::ExitCode;

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

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl OcrError {
    pub fn report(self) -> std::process::ExitCode {
        match self {
            OcrError::Ort( message ) => eprintln!("{message}"),
            OcrError::Image( message ) => eprintln!("{message}"),
            OcrError::DeviceUnavailable( message ) => eprintln!("{message}"),
            OcrError::ModelOutputError( message ) => eprintln!("{message}"),
            OcrError::FeatureDisabled( message ) => eprintln!("{message}"),
            OcrError::InvalidArgument( message ) => eprintln!("{message}"),
            OcrError::Io( message ) => eprintln!("{message}"),
        }
        ExitCode::FAILURE
    }
}
