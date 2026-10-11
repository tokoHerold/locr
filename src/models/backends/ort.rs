use std::sync::LazyLock;

use ndarray::{ArrayView, Dimension};
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};

use crate::{
    Device, ModelCache, OcrError,
    core::traits::{Model, ModelOutput},
};

static CACHE: LazyLock<ModelCache> = LazyLock::new(|| ModelCache::new());

/// ONNX Runtime session wrapper
pub struct OrtModel {
    session: Session,
}

impl OrtModel {
    /// Initializes the model from bytes and optimizes it for the specified [`Device`]
    ///
    /// # Errors
    ///
    /// Returns [`OcrError`] if the ONNX session cannot be initialized on the device.
    pub fn new(
        device: Device,
        model_name: &str,
        model_bytes: &'static [u8],
    ) -> Result<Self, OcrError> {
        let session = CACHE.load_session(model_name, model_bytes, device)?;
        Ok(Self { session })
    }
}

/// Borrowed output buffer referencing tensor memory inside the active session run.
pub struct OrtOutput<'a> {
    outputs: SessionOutputs<'a>,
    shape: Vec<usize>, // Convenience wrapper, for the sake of readability
}

impl<'a> ModelOutput for OrtOutput<'a> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn raw_data(&self) -> &[f32] {
        let (_, slice) = self.outputs[0].try_extract_tensor::<f32>().unwrap();
        slice
    }
}

impl<D: Dimension> Model<D> for OrtModel {
    type Output<'a>
        = OrtOutput<'a>
    where
        Self: 'a;

    fn forward<'a>(
        &'a mut self,
        input: ArrayView<'a, f32, D>,
    ) -> Result<Self::Output<'a>, OcrError> {
        let dyn_view = input.into_dyn(); // low overhead lifetime hack to get rid of D
        let tensor_ref = TensorRef::from_array_view(dyn_view)?;
        let outputs = self.session.run(inputs!["x" => tensor_ref])?;
        let (ort_shape, _) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let shape = ort_shape.iter().map(|&dim| dim as usize).collect(); // i64 -> usize

        Ok(OrtOutput { outputs, shape })
    }
}
