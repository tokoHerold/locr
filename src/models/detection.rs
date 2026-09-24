use std::cmp::max;

use image::DynamicImage;
use ndarray::Array4;
use ort::session::SessionOutputs;

use crate::detector::{BoundingBox, Detector};

pub struct PaddleDetector {
    /// Path to the model
    pub model_path: String,
    /// Minimum certainty of pixels to be recognized as text; range in [0.0, 1.0]
    pub detection_threshold: f32,
    /// Minimum confidence for an entire bounding box to count as text; range in [0.0, 1.0]
    pub box_threshold: f32,
    /// Enlargement factor of detected bounding boxes
    pub unclip_ratio: f32,
    /// Maximum side length of an image. Must be a multiple of 32.
    /// Larger values will capture small text, but take more memory (O(n^2)).
    pub limit_side_length: u32,

    /// Image normalization mean
    normalization_mean: [f32; 3],
    /// Image normalization standard deviation
    normalization_std: [f32; 3],
}

impl PaddleDetector {
    pub fn new(model_path: &str) -> Self {
        Self {
            model_path: model_path.to_string(),
            detection_threshold: 0.3,
            box_threshold: 0.5,
            unclip_ratio: 1.6,
            limit_side_length: 960,
            normalization_mean: [0.485, 0.456, 0.406],
            normalization_std: [0.229, 0.224, 0.225],
        }
    }
}

impl Detector for PaddleDetector {
    fn preprocess(&self, image: &DynamicImage) -> Array4<f32> {
        let image_width = image.width();
        let image_height = image.height();

        // Clip image dimensions to maximum_side_length
        let downscale_ratio =
            if image_width > self.limit_side_length || image_height > self.limit_side_length {
                f64::from(self.limit_side_length) / f64::from(max(image_width, image_height))
            } else {
                1.0
            };
        let resize = |x: u32| -> u32 {
            ((downscale_ratio * f64::from(x) / 32.0).ceil() * 32.0).max(32.0) as u32
        };
        let resized_width = resize(image_width);
        let resized_height = resize(image_height);
        let resized_image = image
            .resize_exact(
                resized_width,
                resized_height,
                image::imageops::FilterType::Nearest,
            )
            .to_rgb8();

        // Store normalized image in tensor
        let mut tensor = Array4::<f32>::zeros((
            1,                       // Batch dimension
            3,                       // RGB
            resized_height as usize, // Image height
            resized_width as usize,  // Image width
        ));
        // Normalized value = (x - mean) / stddev
        let normalize = |pixel: &image::Rgb<u8>, idx: usize| -> f32 {
            (((pixel[idx] as f32) / 255.0) - self.normalization_mean[idx])
                / self.normalization_std[idx]
        };
        for (x, y, pixel) in resized_image.enumerate_pixels() {
            tensor[[0, 0, y as usize, x as usize]] = normalize(pixel, 0);
            tensor[[0, 1, y as usize, x as usize]] = normalize(pixel, 1);
            tensor[[0, 2, y as usize, x as usize]] = normalize(pixel, 2);
        }

        tensor
    }

    fn postprocess(&self, model_output: &SessionOutputs) -> Vec<BoundingBox> {
        todo!()
    }

    fn infer(&self, input: &Array4<f32>) -> SessionOutputs<'_> {
        todo!()
    }
}
