use std::{
    env,
    path::Path,
    process::{Command, ExitCode},
};

fn main() -> ExitCode {
    let repo_root = env!("CARGO_MANIFEST_DIR");
    let script = Path::new(repo_root).join("ci/examples_perf.py");
    let status = Command::new("python3")
        .arg(script)
        .args(env::args_os().skip(1))
        .current_dir(repo_root)
        .status()
        .expect("failed to run python3 ci/examples_perf.py");

    ExitCode::from(status.code().unwrap_or(1) as u8)
}
