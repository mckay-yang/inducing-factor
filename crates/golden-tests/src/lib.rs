//! Integration and end-to-end tests for the inducing_factor Rust
//! rewrite. Uses synthetic data generators to validate the full
//! pipeline without requiring real geospatial datasets.
//!
//! ## Modules
//!
//! - `fixtures`: path resolution for golden baseline files in
//!   `tests/golden/`.
//! - `golden_tests`: reusable test helpers that load golden fixtures
//!   and run comparisons against Rust outputs.

pub mod fixtures;
pub mod golden_tests;

pub use fixtures::*;
pub use golden_tests::*;
