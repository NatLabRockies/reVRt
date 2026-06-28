//! Command line support for RevX-Transmission

use std::path::PathBuf;

use clap::Parser;
use tracing::{debug, info, trace};

use revrt::resolve_with_routing_options;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser)]
#[command(version, about, author, long_about = None)]
struct Cli {
    #[arg(short, long, action=clap::ArgAction::Count)]
    verbose: u8,

    #[arg(short, long, value_name = "DATASET")]
    dataset: PathBuf,

    #[arg(long = "cost-function", value_name = "COST_FUNCTION")]
    cost_function: String,

    #[arg(long = "start", value_delimiter = ',', value_name = "START")]
    start: Vec<usize>,

    #[arg(long = "end", value_delimiter = ',', value_name = "END")]
    end: Vec<usize>,

    #[arg(long = "cache-size", value_name = "CACHE_SIZE")]
    cache_size: Option<usize>,
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let cli = Cli::parse();

    let tracing_level = match cli.verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .with_thread_ids(true)
        .init();
    debug!("Verbose level: {}", cli.verbose);

    trace!("User given dataset: {:?}", cli.dataset);

    assert_eq!(cli.start.len(), 2);
    let start = revrt::ArrayIndex::new_ij(cli.start[0] as u64, cli.start[1] as u64);
    trace!("Starting point: {:?}", start);

    assert_eq!(cli.end.len(), 2);
    let end = vec![revrt::ArrayIndex::new_ij(
        cli.end[0] as u64,
        cli.end[1] as u64,
    )];
    trace!("Ending point: {:?}", end);

    let result = resolve_with_routing_options(
        cli.dataset,
        &cli.cost_function,
        "dijkstra",
        &[start],
        end,
        None,
        250_000_000,
    )
    .unwrap();
    println!("Results: {:?}", result);
    info!("Final solutions: {:?}", result);
}
