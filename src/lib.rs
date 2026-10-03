pub mod core;
pub mod models;
pub mod pipeline;

pub use core::{
    BoundingBox, DetectionResult, Device, OcrEngine, OcrError, OcrResult, RecognitionResult,
    TextDetector, TextRecognizer,
};
pub use models::{PaddleDetector, PaddleRecognizer};
pub use pipeline::TwoStagePipeline;

/// PaddleOCR v6 engine
pub type PaddleOcr = TwoStagePipeline<PaddleDetector, PaddleRecognizer>;

/// Instantiates the default bundled PaddleOCR pipeline on the requested hardware device.
///
/// Models and character dictionaries are loaded directly from embedded memory with zero disk access.
///
/// # Errors
///
/// Returns [`OcrError`] if model sessions cannot be initialized on the device.
pub fn default_engine(device: Device, max_batch_size: usize) -> Result<PaddleOcr, OcrError> {
    let detector = PaddleDetector::new(device)?;
    let recognizer = PaddleRecognizer::new(device, max_batch_size)?;
    Ok(PaddleOcr::new(detector, recognizer))
}
