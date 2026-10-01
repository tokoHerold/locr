use std::cmp::max;

use image::{GrayImage, Luma, RgbImage, imageops::resize};
use imageproc::region_labelling::{Connectivity, connected_components};
use ndarray::Array4;
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};

use crate::core::{traits::TextDetector, types::{BoundingBox, DetectionResult}};

pub struct PaddleDetector {
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

#[derive(Debug, Clone)]
struct TextArea {
    x_min: u32,
    x_max: u32,
    y_min: u32,
    y_max: u32,
    pixel_count: usize,
    score_sum: f32,
}

impl PaddleDetector {
    pub fn new() -> Self {
        Self {
            detection_threshold: 0.3,
            box_threshold: 0.5,
            unclip_ratio: 1.6,
            limit_side_length: 960,
            normalization_mean: [0.485, 0.456, 0.406],
            normalization_std: [0.229, 0.224, 0.225],
        }
    }
}

impl TextDetector for PaddleDetector {
    fn preprocess(&self, image: &RgbImage) -> Array4<f32> {
        let image_width = image.width();
        let image_height = image.height();

        // Clip image dimensions to maximum_side_length
        let downscale_ratio =
            if image_width > self.limit_side_length || image_height > self.limit_side_length {
                f64::from(self.limit_side_length) / f64::from(max(image_width, image_height))
            } else {
                1.0
            };
        let scale = |x: u32| -> u32 {
            ((downscale_ratio * f64::from(x) / 32.0).ceil() * 32.0).max(32.0) as u32
        };
        let resized_width = scale(image_width);
        let resized_height = scale(image_height);
        let resized_image = resize(
                image,
                resized_width,
                resized_height,
                image::imageops::FilterType::Nearest,
            );

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

    fn postprocess(
        &self,
        model_output: &SessionOutputs,
        image: &RgbImage,
    ) -> Vec<DetectionResult> {
        // Extract shape & data from model output
        let (shape, output_value) = model_output[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract model output");
        let height = shape[2] as usize;
        let width = shape[3] as usize;

        // Create binary mask for image: filter out low probabilities
        let binary_pixels: Vec<u8> = output_value
            .iter()
            .map(|&probability| {
                if probability > self.detection_threshold {
                    255 // Text detected
                } else {
                    0 // No text detected
                }
            })
            .collect();
        let binary_mask = GrayImage::from_raw(width as u32, height as u32, binary_pixels)
            .expect("Failed to allocate grayscale image");

        // Extract connected components from heatmap
        let mut text_areas: Vec<Option<TextArea>> = vec![None; 128];
        // Labels each connected region with a unique number
        let labeled_image = connected_components(&binary_mask, Connectivity::Eight, Luma([0u8]));
        const BACKGROUND: Luma<u32> = Luma([0u32]);
        for ((x, y, &label), &score) in labeled_image.enumerate_pixels().zip(output_value) {
            if label == BACKGROUND {
                continue; // No text in pixel
            }

            let label = label.0[0] as usize; // Extract integer out of Luma struct
            while label > text_areas.len() {
                let space = (label + 63) & !63; // Round up to next multiple of 64
                text_areas.resize_with(space, || None); // Ensure vec is big enough
            }
            if let Some(text_area) = &mut text_areas[label] {
                text_area.score_sum += score;
                text_area.pixel_count += 1;
                if x < text_area.x_min {
                    text_area.x_min = x;
                }
                if x > text_area.x_max {
                    text_area.x_max = x;
                }
                text_area.y_max = y;
            } else {
                text_areas[label] = Some(TextArea {
                    x_min: x,
                    x_max: x,
                    y_max: y,
                    y_min: y,
                    pixel_count: 1,
                    score_sum: score,
                })
            }
        }

        // Scale factor: Original image vs inference output
        let original_width = image.width();
        let original_height = image.height();
        let scale_x = original_width as f32 / width as f32;
        let scale_y = original_height as f32 / height as f32;

        // Convert extracted text areas into bounding boxes
        text_areas
            .iter()
            .flatten()
            .filter_map(|text_area| -> Option<DetectionResult> {
                // Filter out results with undesirable confidence
                let width = text_area.x_max - text_area.x_min + 1;
                let height = text_area.y_max - text_area.y_min + 1;
                let average_score = text_area.score_sum / text_area.pixel_count as f32;
                if average_score < self.box_threshold || width < 4 || height < 4 {
                    return None;
                }

                // Unclipping: Grow AABB in all directions
                let area = (width * height) as f32;
                let perimter = (2 * (width + height)) as f32;
                let growth_distance = area * self.unclip_ratio / perimter;

                // Rescale tensor boxes to original image
                let x1 = (((text_area.x_min as f32 - growth_distance) * scale_x).round() as u32)
                    .clamp(0, original_width);
                let x2 = (((text_area.x_max as f32 + growth_distance) * scale_x).round() as u32)
                    .clamp(0, original_width);
                let y1 = (((text_area.y_min as f32 - growth_distance) * scale_y).round() as u32)
                    .clamp(0, original_height);
                let y2 = (((text_area.y_max as f32 + growth_distance) * scale_y).round() as u32)
                    .clamp(0, original_height);
                Some(DetectionResult::new(
                    BoundingBox::new(x1, y1, x2 - x1, y2 - y1),
                    average_score,
                ))
            })
            .collect()
    }

    fn infer<'a>(&self, session: &'a mut Session, input: &Array4<f32>) -> SessionOutputs<'a> {
        let tensor = TensorRef::from_array_view(input.view()).unwrap();
        session
            .run(inputs!["x" => tensor])
            .expect("An error occured during detection inference.")
    }
}
