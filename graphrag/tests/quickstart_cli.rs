#![cfg(feature = "cli")]

use std::fs;
use std::process::Command;

#[test]
fn cli_quickstart_indexes_and_reopens_workspace() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let document = temp.path().join("document.txt");
    let workspace = temp.path().join("workspace");
    fs::write(
        &document,
        include_str!("../../examples/quickstart/document.txt"),
    )
    .expect("write UTF-8 document");

    let index = Command::new(env!("CARGO_BIN_EXE_graphrag"))
        .args([
            "index",
            document.to_str().expect("UTF-8 document path"),
            "--workspace",
            workspace.to_str().expect("UTF-8 workspace path"),
        ])
        .current_dir(temp.path())
        .output()
        .expect("run index command");
    assert!(
        index.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&index.stderr)
    );

    let ask = Command::new(env!("CARGO_BIN_EXE_graphrag"))
        .args([
            "ask",
            "Who maintains the Prisma project?",
            "--workspace",
            workspace.to_str().expect("UTF-8 workspace path"),
        ])
        .current_dir(temp.path())
        .output()
        .expect("run ask command in a new process");
    assert!(
        ask.status.success(),
        "ask failed: {}",
        String::from_utf8_lossy(&ask.stderr)
    );
    let output = String::from_utf8_lossy(&ask.stdout);
    assert!(
        output.contains("Maria Rossi"),
        "unexpected answer: {output}"
    );
}

#[test]
fn cli_turnkey_commands_reject_invalid_config() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let document = temp.path().join("document.txt");
    let config = temp.path().join("invalid.toml");
    let workspace = temp.path().join("workspace");
    let invalid_workspace = temp.path().join("invalid-workspace");
    fs::write(
        &document,
        "The Prisma project is maintained by Maria Rossi.",
    )
    .expect("write document");
    fs::write(&config, "[retrieval]\ntop_k = 0\n").expect("write invalid config");

    let setup = Command::new(env!("CARGO_BIN_EXE_graphrag"))
        .args([
            "index",
            document.to_str().expect("UTF-8 document path"),
            "--workspace",
            workspace.to_str().expect("UTF-8 workspace path"),
        ])
        .current_dir(temp.path())
        .output()
        .expect("run setup index command");
    assert!(
        setup.status.success(),
        "setup index failed: {}",
        String::from_utf8_lossy(&setup.stderr)
    );

    let index = Command::new(env!("CARGO_BIN_EXE_graphrag"))
        .args([
            "--config",
            config.to_str().expect("UTF-8 config path"),
            "index",
            document.to_str().expect("UTF-8 document path"),
            "--workspace",
            invalid_workspace.to_str().expect("UTF-8 workspace path"),
        ])
        .current_dir(temp.path())
        .output()
        .expect("run index command");
    assert!(
        !index.status.success(),
        "invalid config was accepted by index"
    );

    let ask = Command::new(env!("CARGO_BIN_EXE_graphrag"))
        .args([
            "--config",
            config.to_str().expect("UTF-8 config path"),
            "ask",
            "Who maintains Prisma?",
            "--workspace",
            workspace.to_str().expect("UTF-8 workspace path"),
        ])
        .current_dir(temp.path())
        .output()
        .expect("run ask command");
    assert!(!ask.status.success(), "invalid config was accepted by ask");
    let ask_stderr = String::from_utf8_lossy(&ask.stderr);
    assert!(
        ask_stderr.contains("top_k")
            || ask_stderr.to_ascii_lowercase().contains("top-k")
            || ask_stderr.to_ascii_lowercase().contains("validation")
            || ask_stderr
                .to_ascii_lowercase()
                .contains("configuration error"),
        "ask failed for an unrelated reason: {ask_stderr}"
    );
}

#[test]
fn cli_index_updates_existing_workspace_without_clearing_documents() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let first = temp.path().join("first.txt");
    let second = temp.path().join("second.txt");
    let workspace = temp.path().join("workspace");
    fs::write(&first, "The Aurora project is maintained by Ada Lovelace.")
        .expect("write first document");
    fs::write(
        &second,
        "The Borealis project is maintained by Grace Hopper.",
    )
    .expect("write second document");

    for document in [&first, &second] {
        let index = Command::new(env!("CARGO_BIN_EXE_graphrag"))
            .args([
                "index",
                document.to_str().expect("UTF-8 document path"),
                "--workspace",
                workspace.to_str().expect("UTF-8 workspace path"),
            ])
            .current_dir(temp.path())
            .output()
            .expect("run index command");
        assert!(
            index.status.success(),
            "index failed: {}",
            String::from_utf8_lossy(&index.stderr)
        );
    }

    let ask = |question: &str| {
        Command::new(env!("CARGO_BIN_EXE_graphrag"))
            .args([
                "ask",
                question,
                "--workspace",
                workspace.to_str().expect("UTF-8 workspace path"),
            ])
            .current_dir(temp.path())
            .output()
            .expect("run ask command")
    };

    let aurora = ask("Who maintains the Aurora project?");
    assert!(
        aurora.status.success(),
        "ask failed: {}",
        String::from_utf8_lossy(&aurora.stderr)
    );
    assert!(
        String::from_utf8_lossy(&aurora.stdout).contains("Ada Lovelace"),
        "first document was lost: {}",
        String::from_utf8_lossy(&aurora.stdout)
    );

    let borealis = ask("Who maintains the Borealis project?");
    assert!(
        borealis.status.success(),
        "ask failed: {}",
        String::from_utf8_lossy(&borealis.stderr)
    );
    assert!(
        String::from_utf8_lossy(&borealis.stdout).contains("Grace Hopper"),
        "second document was not indexed: {}",
        String::from_utf8_lossy(&borealis.stdout)
    );
}
