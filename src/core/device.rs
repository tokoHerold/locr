use ort::session::builder::SessionBuilder;

use crate::core::error::OcrError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Standard CPU execution provider. Possibly slowest, but guaranteed to exist.
    #[default]
    Cpu,
    /// NVIDIA CUDA execution provider, with specified GPU device index.
    Cuda { device_id: i32 },
    /// Microsoft DirectML execution provider (cross-vendor Windows/WSL2).
    DirectMl { device_id: i32 },
    /// Apple Silicon CoreML execution provider.
    CoreMl,
}

impl Device {
    /// Returns a short unique identifier for cache file naming.
    pub fn get_identifier(&self) -> &str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda { .. } => "cuda",
            Device::DirectMl { .. } => "directml",
            Device::CoreMl => "coreml",
        }
    }

    pub fn configure_session(&self, builder: SessionBuilder) -> Result<SessionBuilder, OcrError> {
        match self {
            Device::Cpu => Ok(builder),
            #[cfg(feature = "cuda")]
            Device::Cuda { device_id } => {
                Ok(builder.with_execution_providers([ort::ep::CUDA::default()
                    .with_device_id(*device_id)
                    .build()])?)
            }

            #[cfg(not(feature = "cuda"))]
            Device::Cuda { .. } => Err(OcrError::FeatureDisabled("cuda".to_string())),
            #[cfg(feature = "directml")]
            Device::DirectMl { device_id } => {
                Ok(
                    builder.with_execution_providers([ort::ep::DirectML::default()
                        .with_device_id(*device_id)
                        .build()])?,
                )
            }
            #[cfg(not(feature = "directml"))]
            Device::DirectMl { .. } => Err(OcrError::FeatureDisabled("directml".to_string())),
            #[cfg(feature = "coreml")]
            Device::CoreMl => Ok(builder.with_execution_providers([[ep::CoreML::default()]])?),
            #[cfg(not(feature = "coreml"))]
            Device::CoreMl => Err(OcrError::FeatureDisabled("coreml".to_string())),
        }
    }
}
