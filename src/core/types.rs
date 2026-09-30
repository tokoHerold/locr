/// An axis-aligned rectangular bounding box in image coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundingBox {
    /// x-coordinate of top-left corner
    pub x: u32,
    /// y-coordinate of top-left corner
    pub y: u32,
    /// Width of the box in pixels
    pub width: u32,
    /// Height of the box in pixels
    pub height: u32
}

/// Pipeline result from a text detection stage
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Axis-aligned bounding box around text
    pub bounding_box: BoundingBox,
    /// Detection confidence score
    pub score: f32,
}

/// Pipeline result from a text recognition stage
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// Decoded text
    pub text: String,
    /// Recognition confidence score
    pub score: f32,
}

/// A recognized text fragment associated with its spatial bounding box and confidences.
#[derive(Debug, Clone)]
pub struct TextItem {
    /// Stores bounding box and detection confidence
    pub detection_result: DetectionResult,
    /// Stores decoded text and recognition confidence
    pub recognition_result: RecognitionResult,
}

/// Output returned by an [`OcrEngine`](crate::core::traits::OcrEngine).
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// Ordered collection of detected and recognized text items.
    pub items: Vec<TextItem>,
    /// End-to-end execution duration in milliseconds.
    pub processing_time_ms: u64,
}
