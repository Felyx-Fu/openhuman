//! Tauri commands for the Claude Code CLI provider.
//!
//! Provides a cross-platform "open a terminal and run `claude auth login`"
//! helper. The CLI's OAuth flow is interactive (it prints a URL and
//! waits for the user to paste a code), so we can't host it in-app — we
//! detach into the user's native terminal so they complete login there,
//! then return to OpenHuman and click Recheck in the settings card.

use std::process::Command;

const CLAUDE_LOGIN_COMMAND: &str = "claude auth login --claudeai";
#[cfg(any(target_os = "linux", test))]
const CLAUDE_LOGIN_ARGS: &[&str] = &["claude", "auth", "login", "--claudeai"];
#[cfg(any(target_os = "linux", test))]
const NO_TERMINAL_ERROR: &str = "no terminal emulator found (tried x-terminal-emulator, gnome-terminal, konsole, xfce4-terminal, xterm). Run `claude auth login --claudeai` manually.";

/// Open the user's native terminal and run `claude auth login` inside it.
///
/// Returns the name of the terminal emulator we launched (for UI
/// confirmation) or an error string if no terminal could be opened.
///
/// Platform behaviour:
///   - Windows: `cmd /c start "" cmd /k claude auth login --claudeai`
///   - macOS:   `osascript` → Terminal.app `do script "claude auth login --claudeai"`
///   - Linux:   try `x-terminal-emulator`, then `gnome-terminal`,
///              `konsole`, `xfce4-terminal`, `xterm` in that order
#[tauri::command]
pub fn claude_code_login_launch() -> Result<String, String> {
    #[cfg(target_os = "windows")]
    {
        // `start ""` opens a new console window; the empty quoted title
        // prevents cmd from interpreting the first arg as a title.
        // `cmd /k` keeps the window open after `claude auth login` exits so
        // the user can read any final output.
        Command::new("cmd")
            .args(["/c", "start", "", "cmd", "/k", CLAUDE_LOGIN_COMMAND])
            .spawn()
            .map_err(|e| format!("failed to open cmd: {e}"))?;
        return Ok("cmd".into());
    }

    #[cfg(target_os = "macos")]
    {
        let script = format!(
            r#"tell application "Terminal"
    activate
    do script "{CLAUDE_LOGIN_COMMAND}"
end tell"#
        );
        Command::new("osascript")
            .arg("-e")
            .arg(script)
            .spawn()
            .map_err(|e| format!("failed to open Terminal.app: {e}"))?;
        Ok("Terminal.app".into())
    }

    #[cfg(target_os = "linux")]
    {
        let terminals: &[(&str, &[&str])] = &[
            (
                "x-terminal-emulator",
                &[
                    "-e",
                    CLAUDE_LOGIN_ARGS[0],
                    CLAUDE_LOGIN_ARGS[1],
                    CLAUDE_LOGIN_ARGS[2],
                    CLAUDE_LOGIN_ARGS[3],
                ],
            ),
            (
                "gnome-terminal",
                &[
                    "--",
                    CLAUDE_LOGIN_ARGS[0],
                    CLAUDE_LOGIN_ARGS[1],
                    CLAUDE_LOGIN_ARGS[2],
                    CLAUDE_LOGIN_ARGS[3],
                ],
            ),
            (
                "konsole",
                &[
                    "-e",
                    CLAUDE_LOGIN_ARGS[0],
                    CLAUDE_LOGIN_ARGS[1],
                    CLAUDE_LOGIN_ARGS[2],
                    CLAUDE_LOGIN_ARGS[3],
                ],
            ),
            ("xfce4-terminal", &["-e", CLAUDE_LOGIN_COMMAND]),
            (
                "xterm",
                &[
                    "-e",
                    CLAUDE_LOGIN_ARGS[0],
                    CLAUDE_LOGIN_ARGS[1],
                    CLAUDE_LOGIN_ARGS[2],
                    CLAUDE_LOGIN_ARGS[3],
                ],
            ),
        ];
        for (term, args) in terminals {
            match Command::new(term).args(*args).spawn() {
                Ok(_) => return Ok(term.to_string()),
                Err(_) => continue,
            }
        }
        Err(NO_TERMINAL_ERROR.into())
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Err("claude_code_login_launch is not supported on this platform".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn login_invocation_uses_current_claude_auth_command() {
        assert_eq!(
            CLAUDE_LOGIN_ARGS,
            &["claude", "auth", "login", "--claudeai"]
        );
        assert_eq!(CLAUDE_LOGIN_COMMAND, CLAUDE_LOGIN_ARGS.join(" "));
        assert!(NO_TERMINAL_ERROR.contains(CLAUDE_LOGIN_COMMAND));
        assert!(!NO_TERMINAL_ERROR.contains("claude login"));
    }
}
