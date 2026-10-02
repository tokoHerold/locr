use std::cmp::{Ordering, max};

use image::{
    GenericImageView, Rgb, RgbImage,
    imageops::{FilterType, resize},
};
use ndarray::{Array4, ArrayView1, ArrayView2};
use ort::{inputs, session::{Session, SessionOutputs}, value::TensorRef};

use crate::{core::{
    device::Device, error::OcrError, model_cache::ModelCache, traits::TextRecognizer, types::{BoundingBox, RecognitionResult},
}};
// Injects `pub static CHARACTER_DICT: [&str; <dict_size> + 2]` generated at build time.
include!(concat!(env!("OUT_DIR"), "/dictionary.rs"));

const MODEL_NAME: &str = "PP_OCRv6_tiny_rec";
const TARGET_HEIGHT: u32 = 48;

/// Embedded PaddlePaddle v6 ONNX model bytes for text recognition.
pub static DETECTION_MODEL_BYTES: &[u8] =
    include_bytes!("../../data/models/PP_OCRv6_tiny_rec.onnx");


pub struct PaddleRecognizer {
    session: Session,
}

/// PaddleOCR text recongizer with CTC decoding
impl PaddleRecognizer {
    /// Initializes the recognizer from bundled model bytes and optimizes it for the specified [`Device`]
    ///
    /// # Errors
    ///
    /// Returns [`OcrError`] if the ONNX session cannot be initialized on the device.
    pub fn new(device: Device) -> Result<Self, OcrError> {
        let cache = ModelCache::new();
        let session = cache.load_session(MODEL_NAME, DETECTION_MODEL_BYTES, device)?;
        Ok(Self { session })
    }
}

impl TextRecognizer for PaddleRecognizer {
    fn recognize(
        &mut self,
        image: &RgbImage,
        bounding_boxes: Vec<BoundingBox>,
    ) -> Vec<RecognitionResult> {
        bounding_boxes
            .iter()
            .map(|bounding_box| -> RecognitionResult {
                let tensor = preprocess(image, bounding_box);
                let tensor_ref = TensorRef::from_array_view(tensor.view()).unwrap();
                let model_output = &self.session
                    .run(inputs!["x" => tensor_ref])
                    .expect("An error occured during recognition inference.");
                decoode(&model_output)
            })
            .collect()
    }
    
}

// -= Helper functions =-

/// Custom crop view with lifetime annotation
struct CropView<'a> {
    img: &'a RgbImage,
    bounding_box: &'a BoundingBox,
}

impl<'a> GenericImageView for CropView<'a> {
    type Pixel = Rgb<u8>;
    fn dimensions(&self) -> (u32, u32) {
        (self.bounding_box.width, self.bounding_box.height)
    }

    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        *self
            .img
            .get_pixel(self.bounding_box.x + x, self.bounding_box.y + y)
    }
}

/// Processes an image for model inference.
///
/// # Arguments
///
/// * `image` - reference to RGB image
/// * `bounding_box` - box to crop image
///
/// # Returns
///
/// A tensor that can be inserted into the input layer of the model.
fn preprocess(image: &RgbImage, bounding_box: &BoundingBox) -> Array4<f32> {
    let crop = CropView {
        img: image,
        bounding_box: bounding_box,
    };
    // Rescale image to fit fixed input height of 48 pixels
    let (width, height) = crop.dimensions();
    let target_width = max(
        16,
        (width as f32 * (TARGET_HEIGHT as f32 / height as f32)).round() as u32,
    );
    let resized_crop = resize(&crop, target_width, TARGET_HEIGHT, FilterType::Nearest);

    // Convert resized crop into tensor
    let mut tensor = Array4::<f32>::zeros((
        1,                      // Batch dimension
        3,                      // RGB
        TARGET_HEIGHT as usize, // Image height
        target_width as usize,  // Image width
    ));
    let normalize = |pixel: &image::Rgb<u8>, idx: usize| -> f32 {
        (((pixel[idx] as f32) / 255.0) - 0.5) / 0.5 // Map [0, 255] -> [-1.0, 1.0]
    };
    for (x, y, pixel) in resized_crop.enumerate_pixels() {
        tensor[[0, 0, y as usize, x as usize]] = normalize(pixel, 0);
        tensor[[0, 1, y as usize, x as usize]] = normalize(pixel, 1);
        tensor[[0, 2, y as usize, x as usize]] = normalize(pixel, 2);
    }
    tensor
}

/// Extracts decoded text and confidence from the model output
///
/// # Arguments
///
/// * `model_output` - Output layer from the model
///
/// # Returns
///
/// A list of bounding boxes around each text segment.
fn decoode(model_output: &SessionOutputs) -> RecognitionResult {
    // Parse Model output [1, T, C] (Batch, Time Step, Class) into [T, C]
    let (shape, output_value) = model_output[0]
        .try_extract_tensor::<f32>()
        .expect("Failed to extract model output");
    assert!(
        shape.len() == 3,
        "Recognition model delivered unexpected output."
    );
    assert_eq!(shape[0], 1, "Batch Dimension was not 1!");
    let ctc_logits =
        ArrayView2::from_shape((shape[1] as usize, shape[2] as usize), &output_value).unwrap();
    let time_steps = shape[1] as usize;
    if time_steps == 0 {
        return RecognitionResult {
            text: String::new(),
            score: 0.0,
        };
    }

    // Static compile-time check to assert CHARACTER_DICT is a char array
    let dict: &[char] = &CHARACTER_DICT;

    // For each time step, extract character index with highest probability
    const CTC_BLANK: usize = 0;
    let (first_idx, probability) =
        argmax(&ctc_logits.row(0)).expect("Class dimension cannot be empty");
    let mut text = String::with_capacity(time_steps); // Number of time steps is upper limit
    let mut score: f32 = 0.0;
    if first_idx != CTC_BLANK && first_idx < dict.len() {
        text.push(dict[first_idx]);
    }

    // Two following indices with the same value decode to only one char
    let mut last_idx: usize = first_idx; // Remember last index
    let mut current_segment_probability: f32 = probability; // Remember highest probability of segment

    for timestep_logits in ctc_logits.rows().into_iter().skip(1) {
        let (character_idx, probability) =
            argmax(&timestep_logits).expect("Class dimension cannot be empty");
        // Decode current character:
        if character_idx == last_idx {
            // Same character index without CTC blank: ignore & update probability
            current_segment_probability = current_segment_probability.max(probability);
        } else {
            // New segment: Commit new character and previous segment probability
            if character_idx != CTC_BLANK && character_idx < dict.len() {
                score += current_segment_probability;
                text.push(dict[character_idx]);
            }
            current_segment_probability = probability;
        }
        last_idx = character_idx;
    }
    if time_steps > 1 {
        score += current_segment_probability; // Commit probability of last segment
    }

    // Calculate final score
    score = if text.len() > 0 {
        score / text.chars().count() as f32
    } else {
        0.0
    };
    RecognitionResult { text, score }
}

/// Returns the maximum value and argmax of an ndarray slice
fn argmax<T: PartialOrd + Copy>(array: &ArrayView1<T>) -> Option<(usize, T)> {
    array
        .iter()
        .enumerate()
        .max_by(|(_idx_a, val_a), (_idx_b, val_b)| {
            val_a.partial_cmp(val_b).unwrap_or(Ordering::Equal)
        })
        .map(|(idx, value)| (idx, *value))
}
