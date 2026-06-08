//! Artifact persistence: serialization/deserialization of trained
//! models, discretizers, factor libraries, and metadata (JSON + binary
//! formats).
//!
//! ## Modules
//!
//! - `artifacts`: typed containers (FactorLibrary, ModelMetadata, …) with
//!   mandatory version/schema metadata.
//! - `golden`: comparison utilities for verifying Rust outputs against
//!   golden baseline fixtures.

pub mod artifacts;
pub mod golden;

pub use artifacts::*;
pub use golden::*;
