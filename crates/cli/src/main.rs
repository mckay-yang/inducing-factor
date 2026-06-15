mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum, ValueHint};

#[derive(Parser)]
#[command(
    name = "if-cli",
    version = "0.1.0",
    about = "CLASS-grid ecosystem disturbance attribution workflow",
    long_about = "Rust-native CLASS-grid workflow: data processing, basin-wide factor mining, and prediction products."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a plan from the JSON CLASS workflow configuration
    PlanWorkflow {
        /// Path to JSON workflow config
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,
    },

    /// Run CLASS-centric data preprocessing from JSON workflow config
    DataProcess {
        /// Path to JSON workflow config
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,
    },

    /// Train the basin-wide model from JSON workflow config
    TrainWorkflow {
        /// Path to JSON workflow config
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,
    },

    /// Predict configured CLASS-grid products from JSON workflow config
    PredictWorkflow {
        /// Path to JSON workflow config
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,
    },

    /// Mine continuous fuzzy interaction factors from cached training features
    MineInteractions {
        /// Path to JSON workflow config
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,
    },

    /// Run JSON workflow data_process -> train -> predict
    RunWorkflow {
        /// Path to JSON workflow config
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,

        /// Reuse existing cache/features and start from train
        #[arg(long)]
        skip_data_process: bool,

        /// Refresh one cache group before running the workflow
        #[arg(long, value_enum)]
        refresh_cache: Option<RefreshCacheArg>,
    },
}

#[derive(Clone, Debug, ValueEnum)]
enum RefreshCacheArg {
    /// Rebuild climate-derived feature caches and downstream model/prediction outputs
    ClimateFeatures,
    /// Rebuild model, interaction summaries, and prediction rasters while preserving feature parquet caches
    ModelOutputs,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::PlanWorkflow { config } => {
            commands::run_plan_workflow(&config);
        }
        Commands::DataProcess { config } => {
            commands::run_data_process_workflow(&config);
        }
        Commands::TrainWorkflow { config } => {
            commands::run_train_workflow(&config);
        }
        Commands::PredictWorkflow { config } => {
            commands::run_predict_workflow(&config);
        }
        Commands::MineInteractions { config } => {
            commands::run_mine_interactions_workflow(&config);
        }
        Commands::RunWorkflow {
            config,
            skip_data_process,
            refresh_cache,
        } => {
            let refresh_cache = refresh_cache.map(|value| match value {
                RefreshCacheArg::ClimateFeatures => commands::WorkflowRefreshCache::ClimateFeatures,
                RefreshCacheArg::ModelOutputs => commands::WorkflowRefreshCache::ModelOutputs,
            });
            commands::run_workflow(&config, skip_data_process, refresh_cache);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn help_lists_workflow_commands_only() {
        let mut cli = Cli::command();
        let help = cli.render_long_help().to_string();

        assert!(help.contains("plan-workflow"));
        assert!(help.contains("data-process"));
        assert!(help.contains("train-workflow"));
        assert!(help.contains("predict-workflow"));
        assert!(help.contains("mine-interactions"));
        assert!(help.contains("run-workflow"));
        assert!(!help.contains("\npipeline"));
        assert!(!help.contains("\nall"));
    }

    #[test]
    fn workflow_commands_parse() {
        let cli = Cli::command();
        for command in [
            "plan-workflow",
            "data-process",
            "train-workflow",
            "predict-workflow",
            "mine-interactions",
        ] {
            let result = cli.clone().try_get_matches_from(vec![
                "cli",
                command,
                "--config",
                "configs/class_basin_workflow.json",
            ]);
            assert!(result.is_ok(), "{command} should parse successfully");
        }

        let run_result = cli.try_get_matches_from(vec![
            "cli",
            "run-workflow",
            "--config",
            "configs/class_basin_workflow.json",
            "--skip-data-process",
            "--refresh-cache",
            "climate-features",
        ]);
        assert!(run_result.is_ok(), "run-workflow should parse successfully");
    }
}
