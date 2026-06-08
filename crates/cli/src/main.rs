mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueHint};

#[derive(Parser)]
#[command(
    name = "cli",
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

    /// Run JSON workflow data_process -> train -> predict
    RunWorkflow {
        /// Path to JSON workflow config
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        config: PathBuf,

        /// Reuse existing cache/features and start from train
        #[arg(long)]
        skip_data_process: bool,
    },
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
        Commands::RunWorkflow {
            config,
            skip_data_process,
        } => {
            commands::run_workflow(&config, skip_data_process);
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
        ]);
        assert!(run_result.is_ok(), "run-workflow should parse successfully");
    }
}
