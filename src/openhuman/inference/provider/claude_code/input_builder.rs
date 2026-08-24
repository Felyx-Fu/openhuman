//! Build the stream-json stdin payload fed to `claude --input-format stream-json`.
//!
//! The CLI consumes one JSON object per line on stdin. Each line looks
//! like:
//!   { "type":"user", "message":{"role":"user","content":[{"type":"text","text":"..."}]} }
//!
//! v1 piping policy:
//! - On a *new* CC session: send every supported history `ChatMessage` so
//!   claude has full context (system message is conveyed via
//!   `--append-system-prompt`, not stdin). Claude Code only accepts `user`
//!   roles in this stream, so prior assistant responses are represented as
//!   clearly labelled user text.
//! - On a `--resume` of an existing CC session: claude already has prior
//!   turns server-side; we only send the last user turn.

use serde_json::{json, Value};

use crate::openhuman::agent::messages::ChatMessage;

const PREVIOUS_ASSISTANT_PREFIX: &str = "[Previous assistant response]\n";

/// Build the bytes to write to claude's stdin. Returns an empty `Vec`
/// when there is nothing to send (caller should abort).
pub fn build_stdin(messages: &[ChatMessage], is_new_session: bool) -> Vec<u8> {
    let mut out = String::new();
    let to_emit: Vec<&ChatMessage> = if is_new_session {
        messages.iter().filter(|m| m.role != "system").collect()
    } else {
        // Resume: only the trailing user turn matters.
        messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .into_iter()
            .collect()
    };

    for msg in to_emit {
        let text = match msg.role.as_str() {
            "user" => msg.content.clone(),
            "assistant" => format!("{PREVIOUS_ASSISTANT_PREFIX}{}", msg.content),
            // CC stdin doesn't accept `system` or `tool` rows. The system
            // prompt is plumbed via `--append-system-prompt`; tool roles
            // belong to the harness, not the CLI's input format.
            _ => continue,
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
        push_json_line(&mut out, &line);
    }

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
    fn new_session_pipes_supported_history_as_user_messages() {
        let history = vec![
            msg("system", "you are helpful"),
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("user", "how are you?"),
        ];
        let bytes = build_stdin(&history, true);
        let s = String::from_utf8(bytes).unwrap();
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines.len(), 3); // system filtered out
        let events: Vec<Value> = lines
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(events
            .iter()
            .all(|event| event["message"]["role"] == "user"));
        assert_eq!(events[0]["message"]["content"][0]["text"], "hi");
        assert_eq!(
            events[1]["message"]["content"][0]["text"],
            "[Previous assistant response]\nhello"
        );
        assert_eq!(events[2]["message"]["content"][0]["text"], "how are you?");
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
