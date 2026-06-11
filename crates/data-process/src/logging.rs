use std::process::Command;

pub fn log_status(message: &str) {
    eprintln!("[{}] {}", timestamp(), message);
}

pub fn timestamp() -> String {
    Command::new("date")
        .arg("+%Y-%m-%d %H:%M:%S")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown-time".to_string())
}

#[macro_export]
macro_rules! status {
    ($($arg:tt)*) => {{
        $crate::logging::log_status(&format!($($arg)*));
    }};
}
