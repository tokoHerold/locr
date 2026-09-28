use std::cmp::max;

use image::{
    DynamicImage, GenericImageView,
    imageops::{FilterType, resize},
};
use ndarray::Array4;
use ort::session::SessionOutputs;

use crate::{
    detector::BoundingBox,
    recongizer::{RecognitionResult, Recognizer},
};

const TARGET_HEIGHT: u32 = 48;

struct PaddleRecognizer {}

use image::{Rgb, RgbImage};

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

impl Recognizer for PaddleRecognizer {
    fn preprocess(&self, image: &RgbImage, bounding_box: &BoundingBox) -> Array4<f32> {
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
        let resized_crop = resize(&crop, TARGET_HEIGHT, target_width, FilterType::Nearest);

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

    fn decoode(&self, model_output: &SessionOutputs) -> RecognitionResult {
        todo!()
    }

    fn infer<'a>(
        &self,
        session: &'a mut ort::session::Session,
        input: &Array4<f32>,
    ) -> SessionOutputs<'a> {
        todo!()
    }
}
