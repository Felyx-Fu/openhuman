//! Build the stream-json stdin payload fed to `claude --input-format stream-json`.
//!
//! The CLI consumes one JSON object per line on stdin. Each line looks
//! like:
//!   { "type":"user", "message":{"role":"user","content":[{"type":"text","text":"..."}]} }
//!
//! v1 piping policy:
//! - On a *new* CC session: serialize every supported history `ChatMessage`
//!   into one user turn so claude has full context without starting a new
//!   generation for each historical row (system message is conveyed via
//!   `--append-system-prompt`, not stdin). Claude Code only accepts `user`
//!   roles in this stream, so prior user turns and assistant responses are
//!   represented as clearly labelled historical context, followed by an
//!   explicitly labelled current user turn.
//! - On a `--resume` of an existing CC session: claude already has prior
//!   turns server-side; we only send the last user turn.

use serde_json::{json, Value};

use crate::openhuman::agent::messages::ChatMessage;

const PREVIOUS_USER_PREFIX: &str = "[Previous user message]\n";
const PREVIOUS_ASSISTANT_PREFIX: &str = "[Previous assistant response]\n";
const CURRENT_USER_PREFIX: &str = "[Current user message]\n";
const HISTORY_SEPARATOR: &str = "\n\n";

/// Build the bytes to write to claude's stdin. Returns an empty `Vec`
/// when there is nothing to send (caller should abort).
pub fn build_stdin(messages: &[ChatMessage], is_new_session: bool) -> Vec<u8> {
    let text = if is_new_session {
        let mut history: Option<String> = None;
        let current_user_index = messages.iter().rposition(|msg| msg.role == "user");
        for (index, msg) in messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| msg.role != "system")
        {
            let part = match msg.role.as_str() {
                "user" => {
                    let prefix = if Some(index) == current_user_index {
                        CURRENT_USER_PREFIX
                    } else {
                        PREVIOUS_USER_PREFIX
                    };
                    format!("{prefix}{}", msg.content)
                }
                "assistant" => format!("{PREVIOUS_ASSISTANT_PREFIX}{}", msg.content),
                // CC stdin doesn't accept `system` or `tool` rows. The system
                // prompt is plumbed via `--append-system-prompt`; tool roles
                // belong to the harness, not the CLI's input format.
                _ => continue,
            };
            if let Some(history) = &mut history {
                history.push_str(HISTORY_SEPARATOR);
                history.push_str(&part);
            } else {
                history = Some(part);
            }
        }
        history
    } else {
        // Resume: only the trailing user turn matters.
        messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
    };

    let Some(text) = text else {
        return Vec::new();
    };

    let line = json!({
        "type": "user",
        "message": {
            // Claude Code's stream-json stdin accepts only user messages,
            // including when prior assistant context is replayed.
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    });
    let mut out = String::new();
    push_json_line(&mut out, &line);

    out.into_bytes()
}

fn push_json_line(buf: &mut String, v: &Value) {
    buf.push_str(&serde_json::to_string(v).unwrap_or_default());
    buf.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        match role {
            "system" => ChatMessage::system(content),
            "user" => ChatMessage::user(content),
            "assistant" => ChatMessage::assistant(content),
            _ => ChatMessage::tool(content),
        }
    }

    #[test]
    fn new_session_serializes_supported_history_as_one_user_message() {
        let history = vec![
            msg("system", "you are helpful"),
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("user", "how are you?"),
        ];
        let bytes = build_stdin(&history, true);
        let s = String::from_utf8(bytes).unwrap();
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines.len(), 1); // system filtered out
        let event: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(event["message"]["role"], "user");
        assert_eq!(
            event["message"]["content"][0]["text"],
            "[Previous user message]\nhi\n\n[Previous assistant response]\nhello\n\n[Current user message]\nhow are you?"
        );
    }

    #[test]
    fn new_session_labels_replayed_user_turns_as_history() {
        let history = vec![
            msg("user", "edit file A"),
            msg("assistant", "done"),
            msg("user", "summarize the changes"),
        ];
        let bytes = build_stdin(&history, true);
        let s = String::from_utf8(bytes).unwrap();
        let event: Value = serde_json::from_str(s.lines().next().unwrap()).unwrap();

        assert_eq!(
            event["message"]["content"][0]["text"],
            "[Previous user message]\nedit file A\n\n[Previous assistant response]\ndone\n\n[Current user message]\nsummarize the changes"
        );
    }

    #[test]
    fn resume_pipes_only_last_user_turn() {
        let history = vec![
            msg("user", "earlier turn"),
            msg("assistant", "earlier reply"),
            msg("user", "follow-up"),
        ];
        let bytes = build_stdin(&history, false);
        let s = String::from_utf8(bytes).unwrap();
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines.len(), 1);
        let event: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(event["message"]["role"], "user");
        assert_eq!(event["message"]["content"][0]["text"], "follow-up");
    }

    #[test]
    fn unsupported_history_roles_are_not_emitted() {
        let history = vec![msg("system", "system"), msg("tool", "tool output")];

        assert!(build_stdin(&history, true).is_empty());
    }

    #[test]
    fn empty_history_yields_empty_bytes() {
        let bytes = build_stdin(&[], true);
        assert!(bytes.is_empty());
    }
}
