//! OpenAI request fields to neutral serving types, and the rejections for
//! everything the runtime does not honour.

use super::types::{ChatRequest, CompletionRequest, Content, Prompt, Stop};
use crate::serve::error::ServeError;
use crate::serve::model::{ChatMessage, GenOpts, Role};

/// Build sampling options from the fields both OpenAI requests share.
///
/// Absent fields keep [`GenOpts::default`], which is greedy decoding — not
/// OpenAI's `temperature=1.0`. A default request here is reproducible.
#[allow(clippy::too_many_arguments)]
pub fn gen_opts(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<&Stop>,
    seed: Option<u64>,
) -> Result<GenOpts, ServeError> {
    let mut opts = GenOpts::default();
    if let Some(value) = max_tokens {
        opts.max_tokens = value;
    }
    if let Some(value) = temperature {
        opts.temperature = value;
    }
    if let Some(value) = top_p {
        opts.top_p = value;
    }
    if let Some(value) = stop {
        opts.stop = value.list();
    }
    opts.seed = seed;
    if opts.max_tokens == 0 {
        return Err(ServeError::bad_request("max_tokens must be at least 1"));
    }
    // Checked here rather than only in the generator so that a streaming
    // request is still rejected with an HTTP status, before any bytes of
    // event stream go out.
    if opts.top_p != 1.0 {
        return Err(ServeError::bad_request(
            "top_p is not supported yet; omit it or send 1.0",
        ));
    }
    if !opts.stop.is_empty() {
        return Err(ServeError::bad_request(
            "stop sequences are not supported yet; use max_tokens instead",
        ));
    }
    Ok(opts)
}

pub fn completion_opts(request: &CompletionRequest) -> Result<GenOpts, ServeError> {
    gen_opts(
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.stop.as_ref(),
        request.seed,
    )
}

pub fn chat_opts(request: &ChatRequest) -> Result<GenOpts, ServeError> {
    gen_opts(
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.stop.as_ref(),
        request.seed,
    )
}

pub fn prompt_text(prompt: Prompt) -> Result<String, ServeError> {
    match prompt {
        Prompt::Text(text) => Ok(text),
        Prompt::Batch(_) => Err(ServeError::bad_request(
            "batched prompts are not supported; send one prompt per request",
        )),
    }
}

pub fn chat_messages(request: &ChatRequest) -> Result<Vec<ChatMessage>, ServeError> {
    if request.messages.is_empty() {
        return Err(ServeError::bad_request("messages must not be empty"));
    }
    request
        .messages
        .iter()
        .map(|message| {
            let role = match message.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => {
                    return Err(ServeError::bad_request(format!(
                        "role {other:?} is not supported; expected system, user or assistant"
                    )));
                }
            };
            let content = match &message.content {
                Content::Text(text) => text.clone(),
                Content::Other(_) => {
                    return Err(ServeError::bad_request(
                        "only string message content is supported; \
                         an array of content parts (or a tool call) has nowhere to go here",
                    ));
                }
            };
            Ok(ChatMessage { role, content })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion(json: &str) -> CompletionRequest {
        serde_json::from_str(json).expect("valid completion request")
    }

    #[test]
    fn a_plain_prompt_round_trips() {
        let request = completion(r#"{"model":"m","prompt":"hello"}"#);
        assert_eq!(prompt_text(request.prompt).unwrap(), "hello");
    }

    #[test]
    fn a_batched_prompt_is_rejected_by_name() {
        let request = completion(r#"{"model":"m","prompt":["a","b"]}"#);
        let error = prompt_text(request.prompt).unwrap_err();
        assert_eq!(
            error.message(),
            "batched prompts are not supported; send one prompt per request"
        );
    }

    #[test]
    fn defaults_are_greedy_and_bounded() {
        let request = completion(r#"{"model":"m","prompt":"x"}"#);
        let opts = completion_opts(&request).unwrap();
        assert_eq!(opts, GenOpts::default());
        assert_eq!(
            opts.temperature, 0.0,
            "a default request must be reproducible"
        );
        assert_eq!(opts.max_tokens, 256);
    }

    #[test]
    fn max_completion_tokens_is_accepted_as_an_alias() {
        let request = completion(r#"{"model":"m","prompt":"x","max_completion_tokens":7}"#);
        assert_eq!(completion_opts(&request).unwrap().max_tokens, 7);
    }

    #[test]
    fn a_zero_token_budget_is_rejected() {
        let request = completion(r#"{"model":"m","prompt":"x","max_tokens":0}"#);
        assert_eq!(
            completion_opts(&request).unwrap_err().message(),
            "max_tokens must be at least 1"
        );
    }

    /// Both spellings parse; what the runtime does not honour is rejected
    /// with a message naming the field, never ignored.
    #[test]
    fn stop_and_top_p_parse_but_are_rejected_by_policy() {
        for json in [
            r#"{"model":"m","prompt":"x","stop":"END"}"#,
            r#"{"model":"m","prompt":"x","stop":["A","B"]}"#,
        ] {
            let error = completion_opts(&completion(json)).unwrap_err();
            assert_eq!(
                error.message(),
                "stop sequences are not supported yet; use max_tokens instead"
            );
        }
        let error =
            completion_opts(&completion(r#"{"model":"m","prompt":"x","top_p":0.9}"#)).unwrap_err();
        assert_eq!(
            error.message(),
            "top_p is not supported yet; omit it or send 1.0"
        );
        // 1.0 is the default, so sending it explicitly must not fail.
        assert!(completion_opts(&completion(r#"{"model":"m","prompt":"x","top_p":1.0}"#)).is_ok());
    }

    /// The raw `Stop` type still maps both shapes to one list — the policy
    /// check lives above it, so this covers the parsing layer alone.
    #[test]
    fn stop_normalises_both_spellings_to_one_list() {
        let one: Stop = serde_json::from_str(r#""END""#).unwrap();
        assert_eq!(one.list(), vec!["END".to_string()]);
        let many: Stop = serde_json::from_str(r#"["A","B"]"#).unwrap();
        assert_eq!(many.list(), vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn unknown_fields_are_ignored_not_fatal() {
        // Clients send n, logprobs, presence_penalty and friends routinely.
        let request =
            completion(r#"{"model":"m","prompt":"x","n":1,"logprobs":true,"presence_penalty":0}"#);
        assert!(completion_opts(&request).is_ok());
    }

    fn chat(json: &str) -> ChatRequest {
        serde_json::from_str(json).expect("valid chat request")
    }

    #[test]
    fn the_three_core_roles_map_across() {
        let request = chat(
            r#"{"model":"m","messages":[
                {"role":"system","content":"be terse"},
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"hello"}]}"#,
        );
        let messages = chat_messages(&request).unwrap();
        assert_eq!(
            messages,
            vec![
                ChatMessage {
                    role: Role::System,
                    content: "be terse".into()
                },
                ChatMessage {
                    role: Role::User,
                    content: "hi".into()
                },
                ChatMessage {
                    role: Role::Assistant,
                    content: "hello".into()
                },
            ]
        );
    }

    #[test]
    fn a_tool_role_is_rejected_by_name() {
        let request = chat(r#"{"model":"m","messages":[{"role":"tool","content":"x"}]}"#);
        let error = chat_messages(&request).unwrap_err();
        assert_eq!(
            error.message(),
            r#"role "tool" is not supported; expected system, user or assistant"#
        );
    }

    #[test]
    fn multimodal_parts_are_rejected_not_dropped() {
        let request = chat(
            r#"{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
        );
        let error = chat_messages(&request).unwrap_err();
        assert!(
            error
                .message()
                .starts_with("only string message content is supported"),
            "{}",
            error
        );
    }

    #[test]
    fn an_empty_message_list_is_rejected() {
        let request = chat(r#"{"model":"m","messages":[]}"#);
        assert_eq!(
            chat_messages(&request).unwrap_err().message(),
            "messages must not be empty"
        );
    }

    /// The untagged `Content` must not swallow a missing field.
    #[test]
    fn a_message_without_content_fails_to_parse() {
        assert!(
            serde_json::from_str::<ChatRequest>(r#"{"model":"m","messages":[{"role":"user"}]}"#)
                .is_err()
        );
    }
}
