//! Concrete model implementations for text detection and recognition.

pub mod ppv6_detection;
pub mod ppv6_recognition;

pub use ppv6_detection::PaddleDetector;
pub use ppv6_recognition::PaddleRecognizer;
