use std::path::PathBuf;


use ort::session::Session;

use crate::core::{device::Device, error::OcrError};

pub struct ModelCache {
    cache_dir: Option<PathBuf>,
}

impl ModelCache {
    /// Constructs a model cache with the operating system's defailt cache directory
    pub fn new() -> Self {
        #[cfg(target_os = "windows")] // %LOCALAPPDATA%\local\locr\cache
        let dir = std::env::var("LOCALAPPDATA").map(|d| PathBuf::from(d).join("locr/cache"));
        #[cfg(not(target_os = "windows"))] // ~/.config/locr/cache
        let dir = std::env::var("XDG_CACHE_HOME")
            .map(|p| PathBuf::from(p).join("locr"))
            .or_else(|_| std::env::var("HOME").map(|p| PathBuf::from(p).join(".cache/locr")));

        Self {
            cache_dir: dir.map_or(None, |d| Some(d))
        }
    }

    pub fn load_session(
        &self,model_name: &str, model_bytes: &'static [u8], device: Device) -> Result<Session, OcrError> {
        todo!("{:?} {:?} {:?}", model_name, model_bytes, device)
    }
}
