use image::imageops::FilterType;
use ndarray::Array4;
use ort::session::Session;

// use crate::ocr::OcrRecognizer;

pub struct CrnnModel {
    /// ONNX Session object
    session: Session,
    /// Mapping from output tensor values to characters
    dictionary: Vec<char>,
    /// Input layer name of the CRNN
    input_name: String,
}

impl CrnnModel {
    /// Creates a new CRNN model for OCR.
    ///
    /// # Arguments
    ///
    /// * `onnx_path` - path to the desired model
    /// * `dict_path` - path to model dictionary
    /// * `input_name` - Name of input layer
    ///
    pub fn new(onnx_path: &str, dict_path: &str, input_name: &str) -> Self {
        let session = Session::builder()
            .unwrap()
            .commit_from_file(onnx_path)
            .expect(&format!("Failed to load ONNX model from path '{:}'", &onnx_path).to_string());
        let dict_string = std::fs::read_to_string(dict_path).expect(&format!("Failed to load model dictionary at '{:}'", &dict_path).to_string());
        let mut dictionary: Vec<char> = dict_string.chars().filter(|c| !c.is_whitespace()).collect();
        dictionary.insert(0, ' ');
        Self { session, dictionary, input_name: input_name.to_string() }
    }
}

// impl OcrRecognizer for CrnnModel {
//     fn preprocess(&self, img: &image::DynamicImage) -> Array4<f32> {
//         let target_width = 320; // TODO read this from somewhere
//         let target_height = 48; // TODO read this from somewhere
//         let mut tensor = Array4::<f32>::zeros((1, 3, target_height as usize, target_width as usize));
//
//         // Clip image down to input layer size
//         let img_resized = img.resize_exact(target_width, target_height, FilterType::Triangle).to_rgb8();
//
//         // Normalize pixel values
//         for (x, y, pixel) in img_resized.enumerate_pixels() {
//             tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32) / 255.0;
//             tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32) / 255.0;
//             tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32) / 255.0;
//         }
//
//         tensor
//     }
//
//     fn decode(&self, output_tensor: ndarray::prelude::ArrayViewD<f32>) -> String {
//         let shape = output_tensor.shape();
//         assert!(shape.len() > 2, "Unexpected result shape");
//         let sequence_length = shape[1];
//         let num_classes = shape[2];
//         let mut result = String::new();
//         todo!()
//     }
//
//     fn session(&self) -> &Session {
//         &self.session
//     }
//
//     fn input_name(&self) -> &str {
//         &self.input_name
//     }
// }
