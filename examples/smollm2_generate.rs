//! Generate real text with SmolLM2-135M, using the HuggingFace GPT-2 tokenizer.
//!
//! ```text
//! cargo run --release --example smollm2_generate -- "Once upon a time"
//! KERNRT_Q8_0=1 cargo run --release --example smollm2_generate -- "The capital of France is"
//! ```

use kernrt::generate::Sampler;
use kernrt::gpt2_tokenizer::Gpt2Tokenizer;
use kernrt::llama::Llama;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/smollm2");
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Once upon a time".to_string());
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);

    let mut model = Llama::from_hf(&dir)?;
    let tokenizer = Gpt2Tokenizer::load(dir.join("tokenizer.json"))?;
    println!(
        "storage {} | vocab {} | prompt {prompt:?}",
        model.storage(),
        tokenizer.vocab_size()
    );

    let prompt_tokens = tokenizer.encode(&prompt, false);
    println!("prompt tokens: {prompt_tokens:?}");

    let mut sampler = Sampler::greedy();
    let start = Instant::now();
    let completion = run_generation(&mut model, &tokenizer, &mut sampler, &prompt_tokens, steps);
    let elapsed = start.elapsed();
    let produced = completion.1;
    println!("\ncompletion:\n{}", completion.0);
    println!(
        "\n{produced} tokens in {:.2}s = {:.1} tok/s",
        elapsed.as_secs_f64(),
        produced as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

/// Feed the prompt, then sample greedily. Returns the decoded text and the
/// number of tokens sampled.
fn run_generation(
    model: &mut Llama,
    tokenizer: &Gpt2Tokenizer,
    sampler: &mut Sampler,
    prompt_tokens: &[usize],
    steps: usize,
) -> (String, usize) {
    let seq_len = model.config().seq_len;
    let mut produced = Vec::new();
    // Feed the prompt first, then keep feeding back the sampled token. Each
    // forward call returns the logits for the *next* token, so the last prompt
    // token is what produces the first sampled one.
    let mut token = *prompt_tokens.first().expect("prompt is non-empty");
    let mut pos = 0;
    while pos < seq_len && produced.len() < steps {
        let logits = model.forward(token, pos);
        pos += 1;
        if pos < prompt_tokens.len() {
            // Still consuming the prompt: the next input is already known.
            token = prompt_tokens[pos];
            continue;
        }
        let next = sampler.sample(logits);
        if next == tokenizer.eos() {
            break;
        }
        produced.push(next);
        token = next;
    }
    (tokenizer.decode_text(&produced), produced.len())
}
