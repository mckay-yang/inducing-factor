mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueHint};

#[derive(Parser)]
#[command(
    name = "if-cli",
    version = "0.1.0",
    about = "Ecosystem Disturbance Attribution CLI — Rust-native pipeline for inducing factor mining",
    long_about = "Analyzes multi-source geospatial data to discover inducing factor combinations\n\
                  (e.g., \"near road + high rainfall + steep slope\") that co-occur with ecological\n\
                  disturbances, then attributes each disturbance to dominant factors."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run chunked feature extraction pipeline
    ///
    /// Reads DEM, climate, landuse, CCDC, road/river data, clips to AOI,
    /// and writes chunked parquet files to the cache directory.
    ///
    /// Examples:
    ///   if-cli pipeline --config configs/example.yaml
    ///   if-cli pipeline --config configs/example.yaml --workers 4
    ///   if-cli pipeline --config configs/example.yaml --dry-run
    Pipeline {
        /// Path to configuration file (YAML)
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,

        /// Override worker count from config
        #[arg(long)]
        workers: Option<usize>,

        /// Validate config and show chunk plan without processing
        #[arg(long)]
        dry_run: bool,
    },

    /// Train attribution model on pipeline cache output
    ///
    /// Fits the AdaptiveDiscretizer, mines inducing factor candidates,
    /// selects top factors, and trains occurrence and class probability models.
    /// Saves all artifacts to the output directory.
    ///
    /// Examples:
    ///   if-cli train --config configs/example.yaml
    ///   if-cli train --config configs/example.yaml --output-dir results/my_model
    Train {
        /// Path to configuration file (YAML)
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,

        /// Override output directory for model artifacts
        #[arg(short = 'O', long, value_hint = ValueHint::DirPath)]
        output_dir: Option<PathBuf>,
    },

    /// Generate annual disturbance probability maps
    ///
    /// Reads chunked feature tables, runs model inference with learned factors,
    /// and writes probability rasters to the output directory.
    ///
    /// Examples:
    ///   if-cli predict --config configs/example.yaml --year 2018
    ///   if-cli predict --config configs/example.yaml --year 2018 --workers 8
    Predict {
        /// Path to configuration file (YAML)
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,

        /// Target year for prediction (default: latest year from config)
        #[arg(short, long)]
        year: Option<i32>,

        /// Override worker count from config
        #[arg(long)]
        workers: Option<usize>,
    },

    /// Run pipeline → train → predict end-to-end
    ///
    /// Executes the full workflow: feature extraction, model training,
    /// and probability map generation in a single command.
    ///
    /// Examples:
    ///   if-cli all --config configs/example.yaml
    ///   if-cli all --config configs/example.yaml --dry-run
    All {
        /// Path to configuration file (YAML)
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,

        /// Validate config and show plan without executing
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pipeline {
            config,
            workers,
            dry_run,
        } => {
            commands::run_pipeline(&config, workers, dry_run);
        }
        Commands::Train { config, output_dir } => {
            commands::run_train(&config, output_dir.as_deref());
        }
        Commands::Predict {
            config,
            year,
            workers,
        } => {
            commands::run_predict(&config, year, workers);
        }
        Commands::All { config, dry_run } => {
            commands::run_all(&config, dry_run);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_help_output() {
        let mut cli = Cli::command();
        let help = cli.render_long_help().to_string();

        assert!(
            help.contains("pipeline"),
            "Help should mention 'pipeline' command"
        );
        assert!(
            help.contains("train"),
            "Help should mention 'train' command"
        );
        assert!(
            help.contains("predict"),
            "Help should mention 'predict' command"
        );
        assert!(help.contains("all"), "Help should mention 'all' command");
    }

    #[test]
    fn test_cli_pipeline_dry_run() {
        // Verify that --dry-run is a recognized flag
        let cli = Cli::command();
        let matches = cli.try_get_matches_from(vec![
            "if-cli",
            "pipeline",
            "--config",
            "configs/example.yaml",
            "--dry-run",
        ]);
        assert!(
            matches.is_ok(),
            "Pipeline with --dry-run should parse successfully"
        );
    }

    #[test]
    fn test_cli_missing_config() {
        let cli = Cli::command();
        let result = cli.try_get_matches_from(vec!["if-cli", "pipeline"]);
        assert!(result.is_err(), "Missing --config should produce an error");
    }

    #[test]
    fn test_cli_worker_override() {
        let cli = Cli::command();
        let result = cli.try_get_matches_from(vec![
            "if-cli",
            "pipeline",
            "--config",
            "configs/example.yaml",
            "--workers",
            "8",
        ]);
        assert!(
            result.is_ok(),
            "Pipeline with --workers should parse successfully"
        );
    }

    #[test]
    fn test_cli_predict_with_year() {
        let cli = Cli::command();
        let matches = cli.try_get_matches_from(vec![
            "if-cli",
            "predict",
            "--config",
            "configs/example.yaml",
            "--year",
            "2020",
        ]);
        assert!(
            matches.is_ok(),
            "Predict with --year should parse successfully"
        );
    }
}
