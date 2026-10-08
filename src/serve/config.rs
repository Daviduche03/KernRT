//! The manifest: which models to load, from where.
//!
//! This is the only module in `serve` that knows a file path or names a
//! network type. Relative paths resolve against the manifest's own directory,
//! so a manifest can be moved together with its weights.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::Registry;
use super::adapters::{LlamaGenerate, NanoJevDecide, YoloDetect};
use super::error::ServeError;
use super::model::{ChatMessage, Model, Role};
use crate::arch::decision::DecisionModel;
use crate::arch::llm::Llama;
use crate::arch::vision::yolo26::Yolo26;
use crate::tokenizer::BpeTokenizer;

/// Detector input size when an entry does not state one: the square
/// Ultralytics exports with `imgsz=640`.
const DEFAULT_INPUT: usize = 640;
/// Decision-head context when an entry does not state one. The checkpoint
/// declares 40960 positions, whose full key/value cache would be ~9 GB;
/// decision leaves are a few hundred tokens.
const DEFAULT_SEQ_LEN: usize = 2048;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub models: Vec<Entry>,
}

/// How to turn chat messages into a prompt.
///
/// KernRT does not interpret Jinja, and guessing a template for a base model
/// silently produces wrong answers — so the manifest supplies one, and
/// [`render`](ChatTemplate::render) refuses to run without it.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatTemplate {
    /// The per-message formats. See [`MessageFormats`].
    pub message: MessageFormats,
    /// Appended after the last message to invite the reply, e.g.
    /// `"Assistant:"`. Not a message, so it has no format of its own.
    #[serde(default)]
    pub generation_prompt: String,
}

/// One string for every role, or one string per role.
///
/// Real templates differ per role — SmolLM2 writes `User: `, ChatML writes
/// `<|user|>` — and a single `{role}` substitution cannot spell both. The
/// string form still covers templates that treat roles uniformly.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageFormats {
    /// Applied to every message, with `{role}` and `{content}` substituted.
    One(String),
    /// One format per role, with `{content}` substituted. A message whose
    /// role has no format is an error rather than a silent omission.
    PerRole {
        #[serde(default)]
        system: Option<String>,
        user: String,
        assistant: String,
    },
}

impl ChatTemplate {
    /// Build the prompt from a conversation.
    ///
    /// Only `{role}` and `{content}` are substituted; there is no expression
    /// language to get subtly wrong.
    pub fn render(&self, messages: &[ChatMessage]) -> Result<String, ServeError> {
        if messages.is_empty() {
            return Err(ServeError::bad_request("messages must not be empty"));
        }
        let mut prompt = String::new();
        for message in messages {
            let format = match &self.message {
                MessageFormats::One(format) => format
                    .replace("{role}", message.role.as_str())
                    .replace("{content}", &message.content),
                MessageFormats::PerRole {
                    system,
                    user,
                    assistant,
                } => {
                    let format = match message.role {
                        Role::System => system.as_deref(),
                        Role::User => Some(user.as_str()),
                        Role::Assistant => Some(assistant.as_str()),
                    };
                    let format = format.ok_or_else(|| {
                        ServeError::bad_request(format!(
                            "the chat template has no format for the {} role; add one under \
                             chat_template.message",
                            message.role.as_str()
                        ))
                    })?;
                    format.replace("{content}", &message.content)
                }
            };
            prompt.push_str(&format);
        }
        prompt.push_str(&self.generation_prompt);
        Ok(prompt)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Entry {
    /// How clients name it: `?model=<id>` or the OpenAI `model` field.
    pub id: String,
    /// `"detect"`, `"decide"`, or `"generate"`.
    pub kind: String,
    pub path: PathBuf,
    /// Detector input size, in pixels. Defaults to 640x640.
    #[serde(default)]
    pub width: Option<usize>,
    #[serde(default)]
    pub height: Option<usize>,
    /// Decision-head context length. Defaults to 2048.
    #[serde(default)]
    pub seq_len: Option<usize>,
    /// Required for `generate` models used through `/v1/chat/completions`.
    /// Omit it and only `/v1/completions` (raw prompts) works.
    #[serde(default)]
    pub chat_template: Option<ChatTemplate>,
}

impl Manifest {
    /// Read and parse `path`, resolving every model path against its directory.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ServeError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| {
            ServeError::bad_request(format!("cannot read manifest {}: {error}", path.display()))
        })?;
        let mut manifest: Manifest = serde_json::from_slice(&bytes).map_err(|error| {
            ServeError::bad_request(format!(
                "{} is not a valid manifest: {error}",
                path.display()
            ))
        })?;
        let base = path.parent().unwrap_or_else(|| Path::new(""));
        for entry in &mut manifest.models {
            if entry.path.is_relative() {
                entry.path = base.join(&entry.path);
            }
        }
        Ok(manifest)
    }

    /// Load every entry. Fails on the first problem rather than serving a
    /// partially-populated registry.
    pub fn registry(&self) -> Result<Registry, ServeError> {
        if self.models.is_empty() {
            return Err(ServeError::bad_request("manifest lists no models"));
        }
        // Validate every id before loading anything: a manifest error must not
        // be masked by a weight file failing to open first.
        let mut seen = std::collections::HashSet::with_capacity(self.models.len());
        for entry in &self.models {
            if entry.id.is_empty() {
                return Err(ServeError::bad_request("a model has an empty id"));
            }
            if !seen.insert(entry.id.as_str()) {
                return Err(ServeError::bad_request(format!(
                    "duplicate model id {:?}",
                    entry.id
                )));
            }
        }

        let mut registry = Registry::new();
        for entry in &self.models {
            registry.register(entry.load()?);
        }
        Ok(registry)
    }
}

impl Entry {
    fn load(&self) -> Result<Model, ServeError> {
        let id = &self.id;
        match self.kind.as_str() {
            "detect" => {
                let width = self.width.unwrap_or(DEFAULT_INPUT);
                let height = self.height.unwrap_or(DEFAULT_INPUT);
                let model = Yolo26::from_onnx(&self.path, width, height).map_err(|error| {
                    ServeError::internal(format!("{id}: cannot load detector: {error}"))
                })?;
                Ok(Model::detect(YoloDetect::new(id.clone(), model)))
            }
            "decide" => {
                let seq_len = self.seq_len.unwrap_or(DEFAULT_SEQ_LEN);
                let model =
                    DecisionModel::from_checkpoint(&self.path, seq_len).map_err(|error| {
                        ServeError::internal(format!("{id}: cannot load decision head: {error}"))
                    })?;
                let tokenizer_path = self.path.join("tokenizer/tokenizer.json");
                let mut tokenizer = BpeTokenizer::load(&tokenizer_path).map_err(|error| {
                    ServeError::internal(format!(
                        "{id}: cannot load {}: {error}",
                        tokenizer_path.display()
                    ))
                })?;
                // The checkpoint's `eos_token_id` is authoritative; tokenizer.json
                // carries no top-level eos field.
                tokenizer.set_eos(model.eos());
                Ok(Model::decide(NanoJevDecide::new(
                    id.clone(),
                    model,
                    tokenizer,
                )))
            }
            "generate" => {
                let model = Llama::from_hf(&self.path).map_err(|error| {
                    ServeError::internal(format!("{id}: cannot load language model: {error}"))
                })?;
                let tokenizer_path = self.path.join("tokenizer.json");
                let tokenizer = BpeTokenizer::load(&tokenizer_path).map_err(|error| {
                    ServeError::internal(format!(
                        "{id}: cannot load {}: {error}",
                        tokenizer_path.display()
                    ))
                })?;
                Ok(Model::generate(LlamaGenerate::new(
                    id.clone(),
                    model,
                    tokenizer,
                    self.chat_template.clone(),
                )))
            }
            other => Err(ServeError::bad_request(format!(
                "unknown model kind {other:?} for {id:?}; expected detect, decide or generate"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_manifest(directory: &Path, body: &str) -> PathBuf {
        std::fs::create_dir_all(directory).unwrap();
        let path = directory.join("models.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn relative_paths_resolve_against_the_manifest() {
        let directory = std::env::temp_dir().join(format!("kernrt-config-{}", std::process::id()));
        let path = write_manifest(
            &directory,
            r#"{"models":[{"id":"a","kind":"detect","path":"weights.onnx"}]}"#,
        );
        let manifest = Manifest::load(&path).unwrap();
        assert_eq!(
            manifest.models[0].path,
            directory.join("weights.onnx"),
            "path should be relative to the manifest, not the process cwd"
        );
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn an_absurd_manifest_is_rejected_before_any_loading() {
        let directory =
            std::env::temp_dir().join(format!("kernrt-config-b-{}", std::process::id()));
        let path = write_manifest(&directory, "not json at all");
        let error = Manifest::load(&path).unwrap_err();
        assert!(
            error.message().contains("is not a valid manifest"),
            "{}",
            error
        );
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn an_empty_manifest_never_serves() {
        let manifest = Manifest { models: Vec::new() };
        assert_eq!(
            manifest
                .registry()
                .err()
                .expect("an empty manifest must be rejected")
                .message(),
            "manifest lists no models"
        );
    }

    #[test]
    fn duplicate_ids_fail_at_load_time_not_at_request_time() {
        let entry = Entry {
            id: "same".into(),
            kind: "detect".into(),
            path: PathBuf::from("missing.onnx"),
            width: None,
            height: None,
            seq_len: None,
            chat_template: None,
        };
        let manifest = Manifest {
            models: vec![entry.clone(), entry],
        };
        let error = manifest
            .registry()
            .err()
            .expect("duplicate ids must be rejected");
        assert_eq!(error.message(), r#"duplicate model id "same""#);
    }

    #[test]
    fn an_unknown_kind_is_rejected() {
        let entry = Entry {
            id: "x".into(),
            kind: "classify".into(),
            path: PathBuf::from("missing.onnx"),
            width: None,
            height: None,
            seq_len: None,
            chat_template: None,
        };
        let manifest = Manifest {
            models: vec![entry],
        };
        let error = manifest
            .registry()
            .err()
            .expect("an unknown kind must be rejected");
        assert!(error.message().contains("unknown model kind"), "{}", error);
    }

    fn one(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
        }
    }

    /// ChatML-style templates treat every role alike.
    #[test]
    fn a_single_format_applies_to_every_role() {
        let template: ChatTemplate = serde_json::from_str(
            r#"{"message":"<|{role}|>\n{content}<|end|>\n","generation_prompt":"<|assistant|>\n"}"#,
        )
        .unwrap();
        let prompt = template
            .render(&[
                one(Role::System, "be terse"),
                one(Role::User, "hi"),
                one(Role::Assistant, "hello"),
            ])
            .unwrap();
        assert_eq!(
            prompt,
            "<|system|>\nbe terse<|end|>\n<|user|>\nhi<|end|>\n\
             <|assistant|>\nhello<|end|>\n<|assistant|>\n"
        );
    }

    /// What SmolLM2's template actually needs: a different literal per role,
    /// which one `{role}` value cannot spell.
    #[test]
    fn a_per_role_format_spells_each_role_differently() {
        let template: ChatTemplate = serde_json::from_str(
            r#"{"message":{
                  "system":"System: {content}\n\n",
                  "user":"User: {content}\n",
                  "assistant":"Assistant: {content}\n"
                },
                "generation_prompt":"Assistant:"}"#,
        )
        .unwrap();
        let prompt = template
            .render(&[one(Role::System, "be terse"), one(Role::User, "the sky")])
            .unwrap();
        assert_eq!(prompt, "System: be terse\n\nUser: the sky\nAssistant:");
    }

    /// A message the template cannot format must fail loudly. Emitting
    /// nothing would hand the model a prompt with a hole in it.
    #[test]
    fn a_role_the_template_omits_is_a_clear_error() {
        let template: ChatTemplate = serde_json::from_str(
            r#"{"message":{"user":"User: {content}\n","assistant":"Assistant: {content}\n"}}"#,
        )
        .unwrap();
        let error = template
            .render(&[one(Role::System, "hi")])
            .expect_err("system has no format");
        assert_eq!(
            error.message(),
            "the chat template has no format for the system role; add one under \
             chat_template.message"
        );
    }

    #[test]
    fn an_empty_conversation_is_rejected() {
        let template: ChatTemplate =
            serde_json::from_str(r#"{"message":"{role}: {content}\n"}"#).unwrap();
        let error = template
            .render(&[])
            .expect_err("an empty conversation must be rejected");
        assert_eq!(error.message(), "messages must not be empty");
    }

    /// `generation_prompt` is optional: some models are happy to just continue.
    #[test]
    fn the_generation_prompt_defaults_to_nothing() {
        let template: ChatTemplate =
            serde_json::from_str(r#"{"message":"{role}: {content}\n"}"#).unwrap();
        assert_eq!(
            template.render(&[one(Role::User, "hi")]).unwrap(),
            "user: hi\n"
        );
    }
}
