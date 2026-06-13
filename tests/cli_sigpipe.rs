#[cfg(unix)]
#[test]
fn help_output_closed_pipe_does_not_panic() {
    use std::process::{Command, Stdio};

    let mut child = Command::new(env!("CARGO_BIN_EXE_openhuman-core"))
        .arg("--help")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn openhuman-core --help");

    drop(child.stdout.take());

    let output = child.wait_with_output().expect("wait for openhuman-core");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stderr.contains("Broken pipe"),
        "stderr must not include a broken-pipe panic: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "stderr must not include a panic report: {stderr}"
    );
}
