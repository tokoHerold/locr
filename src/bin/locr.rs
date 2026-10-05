use std::{
    path::PathBuf,
    process::{ExitCode, exit},
};

use clap::Parser;
use image::ImageReader;
use locr::{Device, ModelCache, OcrEngine, OcrError, default_engine};

#[derive(Parser, Debug)]
#[command(
    name = "locr",
    version,
    about = "Optical Character Recogntion (OCR) CLI for text extraction from images"
)]

struct Cli {
    /// Path to input image
    #[arg(required_unless_present = "clear_cache")]
    image_path: Option<PathBuf>,

    /// Target execution device: cpu, cuda, directml, coreml
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Clear the local model cache files
    #[arg(long)]
    clear_cache: bool,
}

fn run() -> Result<(), OcrError> {
    let args = Cli::parse();

    if args.clear_cache {
        let cache = ModelCache::new();
        cache.flush()?;
        println!("Successfully cleared model cache.");
        if args.image_path.is_none() {
            return Ok(());
        }
    }

    let image_path = args.image_path.as_ref().unwrap();
    let device = Device::from_str(&args.device)?;
    let img = ImageReader::open(image_path);
    if let Err(error) = img {
        eprintln!("Failed to open image: {}", error.to_string());
        exit(1);
    }

    let img = img.unwrap().decode()?;

    let mut engine = default_engine(device, 8)?;
    let result = engine.process(&img)?;

    println!("\n{}", "=".repeat(92));
    println!(
        "{:<3} | {:<28} | {:<9} | {:<9} | Recognized Text",
        "#", "Bounding Box (x, y, w, h)", "Det Score", "Rec Score"
    );
    println!("{}", "-".repeat(92));
    for (idx, (detection_result, recognition_result)) in result.items.iter().enumerate() {
        let bbox = &detection_result.bounding_box;
        let box_str = format!("({}, {}, {}, {})", bbox.x, bbox.y, bbox.width, bbox.height);
        println!(
            "{:<3} | {:<28} | {:<9.4} | {:<9.4} | {}",
            idx + 1, box_str, detection_result.score, recognition_result.score, recognition_result.text
        );
    }
    println!("{}", "=".repeat(92));
    println!("Processed {} text items in {} ms.", result.items.len(), result.processing_time_ms);
    // println!("Hello, world!");
    Ok(())
}

fn main() -> ExitCode {
    return if let Err(ocr_error) = run() {
        ocr_error.report()
    } else {
        ExitCode::SUCCESS
    };
}
