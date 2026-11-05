//! The real binary: socket discovery, and bearer failures before it serves.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const SECRET: &str = "SENTINEL-launch-bearer-do-not-print";

#[test]
fn two_launches_each_print_a_live_socket() {
    for _ in 0..2 {
        let mut child = spawn(&[], &["HIPFIRE_BEARER_FILE"]);
        let line = first_line(&mut child);
        let path = Path::new(line.trim());
        assert!(
            path.exists(),
            "first stdout line is not an existing path: {line}"
        );
        assert!(
            child.try_wait().unwrap().is_none(),
            "process exited after printing the socket"
        );
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[test]
fn missing_bearer_file_exits_before_the_socket_line() {
    let missing = std::env::temp_dir().join(format!(
        "hipfire-provider-missing-bearer-{}",
        std::process::id()
    ));
    let _ = fs::remove_file(&missing);
    let (stdout, stderr, code) =
        run_until_exit(&[("HIPFIRE_BEARER_FILE", &missing.display().to_string())]);
    assert_ne!(code, 0);
    assert!(!stdout_is_socket(&stdout), "stdout: {stdout}");
    assert!(stderr.contains("missing"), "{stderr}");
    assert!(stderr.contains(&missing.display().to_string()), "{stderr}");
    assert!(!stderr.contains(SECRET));
    assert!(!stdout.contains(SECRET));
}

#[test]
fn unreadable_bearer_file_exits_without_echoing_the_secret() {
    let path = bearer_file();
    let mut perms = fs::metadata(&path).unwrap().permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o0);
    }
    fs::set_permissions(&path, perms).unwrap();
    assert!(
        fs::read_to_string(&path).is_err(),
        "mode 000 should hide the bearer file"
    );

    let (stdout, stderr, code) =
        run_until_exit(&[("HIPFIRE_BEARER_FILE", &path.display().to_string())]);
    restore(&path);
    assert_ne!(code, 0);
    assert!(!stdout_is_socket(&stdout), "stdout: {stdout}");
    assert!(stderr.contains("unreadable"), "{stderr}");
    assert!(
        !stderr.contains(SECRET),
        "stderr contained the bearer token"
    );
    assert!(
        !stdout.contains(SECRET),
        "stdout contained the bearer token"
    );
}

#[test]
fn a_readable_bearer_does_not_appear_on_startup() {
    let path = bearer_file();
    let mut child = spawn(&[("HIPFIRE_BEARER_FILE", &path.display().to_string())], &[]);
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let (line_tx, line_rx) = std::sync::mpsc::channel();
    let stdout_task = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut first = String::new();
        reader.read_line(&mut first).expect("socket line");
        let _ = line_tx.send(first);
        let mut rest = Vec::new();
        let _ = reader.read_to_end(&mut rest);
        rest
    });
    let stderr_task = thread::spawn(move || read_to_end(stderr));
    let line = line_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("timed out waiting for the socket path");
    let socket = Path::new(line.trim());
    assert!(socket.exists(), "first stdout line is not an existing path");
    assert!(child.try_wait().unwrap().is_none());
    let _ = child.kill();
    let _ = child.wait();
    let rest = stdout_task.join().unwrap();
    let stderr = stderr_task.join().unwrap();
    let stdout = format!("{line}{}", String::from_utf8_lossy(&rest));
    let stderr = String::from_utf8_lossy(&stderr);
    assert!(
        !stdout.contains(SECRET),
        "stdout contained the bearer token"
    );
    assert!(
        !stderr.contains(SECRET),
        "stderr contained the bearer token"
    );
    restore(&path);
}

fn spawn(set: &[(&str, &str)], remove: &[&str]) -> Child {
    let mut command = Command::new(env!("CARGO_BIN_EXE_infinity-provider-hipfire"));
    command
        .env("HIPFIRE_BASE_URL", "http://127.0.0.1:11435/v1")
        .env_remove("RUST_LOG")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in remove {
        command.env_remove(name);
    }
    for (name, value) in set {
        command.env(name, value);
    }
    command.spawn().unwrap()
}

fn first_line(child: &mut Child) -> String {
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let result = reader.read_line(&mut line);
        let _ = tx.send((result, line, reader));
    });
    let (result, line, _reader) = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("timed out waiting for the socket path");
    result.expect("failed to read stdout");
    assert!(
        line.ends_with('\n'),
        "stdout did not produce a line: {line}"
    );
    line
}

fn run_until_exit(set: &[(&str, &str)]) -> (String, String, i32) {
    let mut child = spawn(set, &[]);
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout_task = thread::spawn(move || read_to_end(stdout));
    let stderr_task = thread::spawn(move || read_to_end(stderr));
    let code = wait_for_exit(&mut child, Duration::from_secs(5));
    let stdout = String::from_utf8_lossy(&stdout_task.join().unwrap()).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_task.join().unwrap()).into_owned();
    (stdout, stderr, code)
}

fn wait_for_exit(child: &mut Child, limit: Duration) -> i32 {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status.code().unwrap_or(1);
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            panic!("process stayed up after a bearer-file failure");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn read_to_end(mut pipe: impl Read) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = pipe.read_to_end(&mut buf);
    buf
}

fn stdout_is_socket(stdout: &str) -> bool {
    stdout.lines().any(|line| Path::new(line).exists())
}

fn bearer_file() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "hipfire-provider-launch-bearer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&path, format!("{SECRET}\n")).unwrap();
    path
}

fn restore(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o644));
    }
    let _ = fs::remove_file(path);
}
