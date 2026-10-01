use std::{fs, path::PathBuf};

use ort::{compiler::ModelCompiler, session::{Session, builder}};

use crate::core::{device::Device, error::OcrError};

pub struct ModelCache {
    cache_dir: Option<PathBuf>,
}

impl ModelCache {
    /// Creates a cache manager using standard operating system cache directories.
    pub fn new() -> Self {
        #[cfg(target_os = "windows")] // %LOCALAPPDATA%\local\locr\cache
        let dir = std::env::var("LOCALAPPDATA").map(|d| PathBuf::from(d).join("locr/cache"));
        #[cfg(not(target_os = "windows"))] // ~/.config/locr/cache
        let dir = std::env::var("XDG_CACHE_HOME")
            .map(|p| PathBuf::from(p).join("locr"))
            .or_else(|_| std::env::var("HOME").map(|p| PathBuf::from(p).join(".cache/locr")));

        Self {
            cache_dir: dir.map_or(None, |d| Some(d)),
        }
    }

    /// Loads an ONNX session for the given model bytes on the target device.
    ///
    /// For CPU execution, loads directly from memory with zero disk overhead.
    /// For hardware EPs, checks local cache for precompiled EP graphs; compiles and caches on first run.
    ///
    /// # Errors
    ///
    /// Returns [`OcrError`] if compilation or session creation fails.
    pub fn load_session(
        &self,
        model_name: &str,
        model_bytes: &'static [u8],
        device: Device,
    ) -> Result<Session, OcrError> {
        // CPU execution does not require EP graph compilation
        if device == Device::Cpu {
            return Ok(Session::builder()?.commit_from_memory(model_bytes)?);
        }

        // Load hardware execution provider
        let mut builder = device.configure_session(Session::builder()?)?;

        // Cache hit: do nohting
        if let Some(cache_dir) = &self.cache_dir {
            let cache_file = cache_dir.join(format!("{}_{}.onnx", model_name, device.get_identifier()));
            if cache_file.exists() && fs::metadata(&cache_file).unwrap().len() > 0 {
                return Ok(builder.commit_from_file(&cache_file)?);
            }
        }

        // Cache miss: compile model
        let compiler = ModelCompiler::new(builder.clone())?.with_model_from_memory(model_bytes)?;
        match &self.cache_dir {
            Some( cache_dir ) => {
                let cache_file = cache_dir.join(format!("{}_{}.onnx", model_name, device.get_identifier()));
                compiler.compile_to_file(&cache_file)?;
                Ok(builder.commit_from_file(&cache_file)?)
            }
            // If cache directory was not found, degrade gracefully to no caching
            None => Ok(builder.commit_from_memory(&compiler.compile_to_buffer()?)?)
        }
    }
}
