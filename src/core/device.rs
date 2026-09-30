use ort::{session::builder::SessionBuilder};

use crate::core::error::OcrError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Standard CPU execution provider. Possibly slowest, but guaranteed to exist.
    #[default]
    Cpu,
    /// NVIDIA CUDA execution provider, with specified GPU device index.
    Cuda,
    /// Microsoft DirectML execution provider (cross-vendor Windows/WSL2).
    DirectMl,
    /// Apple Silicon CoreML execution provider.
    CoreMl,
}

impl Device {
    pub fn configure_session(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder, OcrError> {
        match self {
            Device::Cpu => Ok(builder),
            #[cfg(feature = "cuda")]
            Device::Cuda =>
                Ok(builder.with_execution_providers([ort::ep::CUDA::default()
                    .build()])?) ,

            
            #[cfg(not(feature = "cuda"))]
            Device::Cuda { .. } => Err(OcrError::DeviceUnavailable(
                "CUDA feature is not enabled".into(),
            )),
            #[cfg(feature = "directml")]
            Device::DirectMl { device_id } => todo!(),
            #[cfg(not(feature = "directml"))]
            Device::DirectMl { .. } => todo!(),
            #[cfg(feature = "coreml")]
            Device::CoreMl => todo!(),
            #[cfg(not(feature = "directml"))]
            Device::CoreMl => todo!(),
        }
    }
}
