//! Serve every model in a manifest over one port: the KernRT-native routes
//! and the OpenAI-compatible ones side by side.
//!
//! ```text
//! cargo run --release --features serve --example serve -- models.json
//! cargo run --release --features serve --example serve -- models.json 127.0.0.1:8080
//!
//! # detector — any of these three, all giving the same detections:
//! curl -s --data-binary @frame.rgb 'http://127.0.0.1:8080/infer?model=yolo26n'
//! curl -s -H 'content-type: image/jpeg' --data-binary @cat.jpg 'http://127.0.0.1:8080/infer?model=yolo26n'
//! curl -s -H 'content-type: application/json' -d '{"image_url":"http://127.0.0.1:8080/cat.jpg"}' 'http://127.0.0.1:8080/infer?model=yolo26n'
//! # decision head:
//! curl -s -H 'content-type: application/json' -d '{"state":"...","instructions":"...","criteria":[{"id":"left","description":"Move left."},{"id":"right","description":"Move right."}]}' http://127.0.0.1:8080/decide
//! # generator, OpenAI-shaped:
//! curl -s -H 'content-type: application/json' -d '{"model":"smollm2","messages":[{"role":"user","content":"hi"}]}' http://127.0.0.1:8080/v1/chat/completions
//! curl -s -N -H 'content-type: application/json' -d '{"model":"smollm2","prompt":"Once upon a time","stream":true}' http://127.0.0.1:8080/v1/completions
//! ```
//!
//! Model paths in the manifest resolve relative to the manifest file.

use kernrt::serve::{Manifest, serve};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let manifest_path = args.next().ok_or("usage: serve MANIFEST.json [ADDR]")?;
    let address = args.next().unwrap_or_else(|| "127.0.0.1:8080".into());

    let registry = Manifest::load(&manifest_path)?.registry()?;
    for model in registry.models() {
        println!("loaded {} ({})", model.id(), model.kind());
    }
    serve(&address, registry)?;
    Ok(())
}
