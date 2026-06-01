use clap::Parser;

#[derive(Parser)]
#[command(name = "if-cli", about = "Inducing factor attribution CLI")]
struct Cli {
    #[arg(short, long, default_value = "World")]
    name: String,
}

fn main() {
    let cli = Cli::parse();
    println!("Hello, {}!", cli.name);
}
