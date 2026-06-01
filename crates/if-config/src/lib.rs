//! Configuration model (paths, pipeline parameters, attribution
//! hyper-parameters) deserialized from YAML/TOML via `serde` and
//! the `config` crate.

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(2 + 2, 4);
    }
}
