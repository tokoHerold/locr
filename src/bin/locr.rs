use std::path::PathBuf;

use clap::Parser;

/// Command-line arguments for the `locr` standalone executable.
#[derive(Parser, Debug)]
#[command(name = "locr", version = "0.1.0", about = "High-performance standalone OCR")]
struct Cli {
    /// Path to input image
    #[arg(required = true)]
    image_path: PathBuf,

    /// Target execution device: cpu, cuda, directml, coreml, auto
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Output results as JSON
    #[arg(long)]
    json: bool,
}


fn main() {
    // const DEVICE: Device = Device::Cpu; // TODO: make configurable
    // let mut engine = default_engine(DEVICE, 8).unwrap();
    // let result = engine.process(&image).unwrap();
    // println!(
    //     "Found {} text blocks in {} ms",
    //     result.items.len(),
    //     result.processing_time_ms
    // );
    // println!("{:?}", result.items);

    // println!("{:?}", results);
    println!("Hello, world!");
}
