//! Core abstractions, data models, device routing, and error types for `locr`.

pub mod device;
pub mod error;
pub mod model_cache;
pub mod traits;
pub mod types;

pub use model_cache::ModelCache;
pub use device::Device;
pub use error::OcrError;
pub use traits::{OcrEngine, TextDetector, TextRecognizer};
pub use types::{BoundingBox, DetectionResult, RecognitionResult, OcrResult};
