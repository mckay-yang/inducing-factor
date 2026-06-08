use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        if ancestor.join("Cargo.toml").exists() {
            let cargo_toml =
                std::fs::read_to_string(ancestor.join("Cargo.toml")).unwrap_or_default();
            if cargo_toml.contains("[workspace]") {
                return ancestor.to_path_buf();
            }
        }
    }
    panic!(
        "Could not find workspace root from {}",
        manifest_dir.display()
    );
}

pub fn golden_dir() -> PathBuf {
    workspace_root().join("tests").join("golden")
}

pub fn golden_path(name: &str) -> PathBuf {
    golden_dir().join(name)
}
