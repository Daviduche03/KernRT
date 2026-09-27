//! Time loading SmolLM2, separated into parse and widen.
use kernrt::arch::llm::Llama;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2");
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let m = Llama::from_hf(&dir)?;
        best = best.min(t.elapsed().as_secs_f64());
        std::hint::black_box(&m);
    }
    println!("from_hf (bf16): {:.1} ms", best * 1e3);
    Ok(())
}
