use std::fs;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::{ArgAction, Parser};
use ndarray::Array3;
#[cfg(unix)]
use pprof::ProfilerGuardBuilder;
use revrt::{ArrayIndex, profiling, resolve_parallel_with_routing_options};
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use zarrs::array::{ArrayBuilder, DataType, FillValue};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;

const DEFAULT_COST_FUNCTION: &str =
    r#"{"routing_options":{"default":{"cost_layers":[{"layer_name":"cost"}]}}}"#;
const PPROF_SAMPLE_FREQUENCY_HZ: i32 = 1_000;
const PPROF_BLOCKLIST: [&str; 4] = ["libc", "libgcc", "pthread", "vdso"];

struct PprofOutputs {
    sample_frequency_hz: Option<i32>,
    flamegraph_available: bool,
    stack_summary_available: bool,
    note: Option<&'static str>,
}

#[cfg(unix)]
struct ActiveProfiler {
    guard: pprof::ProfilerGuard<'static>,
}

#[cfg(not(unix))]
struct ActiveProfiler;

impl ActiveProfiler {
    #[cfg(unix)]
    fn start() -> Result<Self, Box<dyn std::error::Error>> {
        let guard = ProfilerGuardBuilder::default()
            .frequency(PPROF_SAMPLE_FREQUENCY_HZ)
            .blocklist(&PPROF_BLOCKLIST)
            .build()?;
        Ok(Self { guard })
    }

    #[cfg(not(unix))]
    fn start() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }

    #[cfg(unix)]
    fn finish(
        self,
        flamegraph_path: &Path,
        stack_report_path: &Path,
    ) -> Result<PprofOutputs, Box<dyn std::error::Error>> {
        let pprof_report = self.guard.report().build()?;
        pprof_report.flamegraph(fs::File::create(flamegraph_path)?)?;
        let stack_report = format!("{pprof_report:?}");
        fs::write(stack_report_path, &stack_report)?;

        Ok(PprofOutputs {
            sample_frequency_hz: Some(PPROF_SAMPLE_FREQUENCY_HZ),
            flamegraph_available: true,
            stack_summary_available: true,
            note: None,
        })
    }

    #[cfg(not(unix))]
    fn finish(
        self,
        _flamegraph_path: &Path,
        _stack_report_path: &Path,
    ) -> Result<PprofOutputs, Box<dyn std::error::Error>> {
        Ok(PprofOutputs {
            sample_frequency_hz: None,
            flamegraph_available: false,
            stack_summary_available: false,
            note: Some(
                "pprof output is unavailable on non-Unix targets in this crate; use Linux or macOS for sampled flamegraphs.",
            ),
        })
    }
}

#[derive(Parser, Debug)]
#[command(version, about = "Generate and profile a synthetic Rust routing case")]
struct Cli {
    #[arg(long, default_value = ".scratch/rust_route_profile")]
    output_dir: PathBuf,

    #[arg(short, long, action = ArgAction::Count, default_value_t = 2)]
    verbose: u8,

    #[arg(long, default_value_t = 5_000)]
    rows: u64,

    #[arg(long, default_value_t = 5_000)]
    cols: u64,

    #[arg(long, default_value_t = 250)]
    chunk_size: u64,

    #[arg(long, default_value_t = 500)]
    distance: u64,

    #[arg(long, default_value = "bidirectional_long_range_dijkstra")]
    algorithm: String,

    #[arg(long, default_value_t = 2_000_000_000)]
    mem_limit_bytes: u64,

    #[arg(long, default_value_t = 1)]
    repeats: u32,

    #[arg(long, default_value_t = 1.0)]
    cost_value: f32,
}

struct SharedFileWriter {
    file: Arc<Mutex<fs::File>>,
}

impl Write for SharedFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut file = self
            .file
            .lock()
            .map_err(|err| io::Error::other(err.to_string()))?;
        file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut file = self
            .file
            .lock()
            .map_err(|err| io::Error::other(err.to_string()))?;
        file.flush()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    validate_inputs(&cli)?;

    let dataset_dir = cli.output_dir.join("dataset.zarr");
    let report_path = cli.output_dir.join("ROUTING_RUST_PROFILE_REPORT.md");
    let log_path = cli.output_dir.join("routing_profile.log");
    let flamegraph_path = cli.output_dir.join("routing_profile_flamegraph.svg");
    let stack_report_path = cli.output_dir.join("routing_profile_stacks.txt");

    fs::create_dir_all(&cli.output_dir)?;
    init_logging(cli.verbose, &log_path)?;
    create_uniform_cost_dataset(
        &dataset_dir,
        cli.rows,
        cli.cols,
        cli.chunk_size,
        cli.cost_value,
    )?;

    let (start, end) = centered_route(cli.rows, cli.cols, cli.distance)?;
    let mut elapsed_runs = Vec::with_capacity(cli.repeats as usize);
    let mut last_solution_count = 0usize;

    profiling::reset();
    profiling::enable();

    let profiler = ActiveProfiler::start()?;

    for _ in 0..cli.repeats {
        let started = Instant::now();
        let (solutions, _routing_options) = resolve_parallel_with_routing_options(
            &dataset_dir,
            DEFAULT_COST_FUNCTION,
            &cli.algorithm,
            std::slice::from_ref(&start),
            vec![end.clone()],
            None,
            cli.mem_limit_bytes,
        )?;
        elapsed_runs.push(started.elapsed());
        last_solution_count = solutions.len();
    }

    profiling::disable();

    let pprof_outputs = profiler.finish(&flamegraph_path, &stack_report_path)?;
    let profile_snapshot = profiling::snapshot();

    let report_inputs = ReportInputs {
        cli: &cli,
        dataset_dir: &dataset_dir,
        start: &start,
        end: &end,
        solution_count: last_solution_count,
        elapsed_runs: &elapsed_runs,
        records: &profile_snapshot,
        log_path: &log_path,
        flamegraph_path: &flamegraph_path,
        stack_report_path: &stack_report_path,
        pprof_outputs: &pprof_outputs,
    };
    write_report(&report_path, &report_inputs)?;

    println!("Dataset: {}", dataset_dir.display());
    println!("Log: {}", log_path.display());
    if pprof_outputs.flamegraph_available {
        println!("Flamegraph: {}", flamegraph_path.display());
    }
    if pprof_outputs.stack_summary_available {
        println!("Stacks: {}", stack_report_path.display());
    }
    println!("Report: {}", report_path.display());

    Ok(())
}

fn init_logging(verbose: u8, log_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tracing_level = match verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_path)?;
    let shared_file = Arc::new(Mutex::new(file));
    let writer = {
        let shared_file = Arc::clone(&shared_file);
        BoxMakeWriter::new(move || SharedFileWriter {
            file: Arc::clone(&shared_file),
        })
    };

    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .with_thread_ids(true)
        .with_ansi(false)
        .with_writer(writer)
        .try_init()
        .map_err(|err| io::Error::other(err.to_string()))?;

    Ok(())
}

fn validate_inputs(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if cli.chunk_size == 0 {
        return Err("chunk-size must be greater than zero".into());
    }
    if cli.rows == 0 || cli.cols == 0 {
        return Err("rows and cols must be greater than zero".into());
    }
    if !cli.rows.is_multiple_of(cli.chunk_size) || !cli.cols.is_multiple_of(cli.chunk_size) {
        return Err("rows and cols must be divisible by chunk-size for this generator".into());
    }
    if cli.distance == 0 {
        return Err("distance must be greater than zero".into());
    }
    if cli.repeats == 0 {
        return Err("repeats must be greater than zero".into());
    }

    Ok(())
}

fn centered_route(
    rows: u64,
    cols: u64,
    distance: u64,
) -> Result<(ArrayIndex, ArrayIndex), Box<dyn std::error::Error>> {
    if distance >= cols {
        return Err("distance must be smaller than the column count".into());
    }
    let half_distance = distance / 2;
    let center_row = rows / 2;
    let center_col = cols / 2;
    if center_col < half_distance || center_col + (distance - half_distance) >= cols {
        return Err("route distance does not fit inside the dataset width".into());
    }

    Ok((
        ArrayIndex::new_ij(center_row, center_col - half_distance),
        ArrayIndex::new_ij(center_row, center_col + (distance - half_distance)),
    ))
}

fn create_uniform_cost_dataset(
    dataset_dir: &Path,
    rows: u64,
    cols: u64,
    chunk_size: u64,
    cost_value: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if dataset_dir.exists() {
        fs::remove_dir_all(dataset_dir)?;
    }

    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(dataset_dir).expect("could not open filesystem store"));

    GroupBuilder::new()
        .build(store.clone(), "/")?
        .store_metadata()?;

    let array = ArrayBuilder::new(
        vec![1, rows, cols],
        vec![1, chunk_size, chunk_size],
        DataType::Float32,
        FillValue::from(zarrs::array::ZARR_NAN_F32),
    )
    .dimension_names(["band", "y", "x"].into())
    .build(store, "/cost")?;

    array.store_metadata()?;

    let values = vec![cost_value; usize::try_from(rows * cols)?];
    let data = Array3::from_shape_vec((1, rows as usize, cols as usize), values)?;
    let subset_start = [0, 0, 0];
    array.store_array_subset_ndarray(&subset_start, data)?;

    Ok(())
}

struct ReportInputs<'a> {
    cli: &'a Cli,
    dataset_dir: &'a Path,
    start: &'a ArrayIndex,
    end: &'a ArrayIndex,
    solution_count: usize,
    elapsed_runs: &'a [std::time::Duration],
    records: &'a [profiling::ProfileRecord],
    log_path: &'a Path,
    flamegraph_path: &'a Path,
    stack_report_path: &'a Path,
    pprof_outputs: &'a PprofOutputs,
}

fn write_report(
    report_path: &Path,
    inputs: &ReportInputs,
) -> Result<(), Box<dyn std::error::Error>> {
    let total_profiled = inputs
        .records
        .iter()
        .map(|record| record.total)
        .sum::<std::time::Duration>();
    let avg_elapsed = std::time::Duration::from_secs_f64(
        inputs
            .elapsed_runs
            .iter()
            .map(std::time::Duration::as_secs_f64)
            .sum::<f64>()
            / inputs.elapsed_runs.len() as f64,
    );

    let mut report = String::new();
    report.push_str("# Rust Routing Profile Report\n\n");
    report.push_str("## Benchmark Setup\n\n");
    report.push_str(&format!("- Dataset: `{}`\n", inputs.dataset_dir.display()));
    report.push_str(&format!(
        "- Grid: `{} x {}`\n",
        inputs.cli.rows, inputs.cli.cols
    ));
    report.push_str(&format!(
        "- Chunk size: `{} x {}`\n",
        inputs.cli.chunk_size, inputs.cli.chunk_size
    ));
    report.push_str(&format!("- Cell value: `{}`\n", inputs.cli.cost_value));
    report.push_str(&format!("- Algorithm: `{}`\n", inputs.cli.algorithm));
    report.push_str(&format!(
        "- Memory limit: `{}` bytes\n",
        inputs.cli.mem_limit_bytes
    ));
    report.push_str(&format!("- Repeats: `{}`\n", inputs.cli.repeats));
    report.push_str(&format!("- Start: `{:?}`\n", inputs.start));
    report.push_str(&format!("- End: `{:?}`\n", inputs.end));
    report.push_str(&format!(
        "- Solutions returned on last run: `{}`\n",
        inputs.solution_count
    ));
    report.push_str(&format!(
        "- Average wall-clock time per run: `{:.3}` s\n",
        avg_elapsed.as_secs_f64()
    ));
    if let Some(sample_frequency_hz) = inputs.pprof_outputs.sample_frequency_hz {
        report.push_str(&format!(
            "- PProf sample frequency: `{}` Hz\n",
            sample_frequency_hz
        ));
    }
    report.push_str(&format!("- Tracing log: `{}`\n", inputs.log_path.display()));
    report.push_str("\n## PProf Outputs\n\n");
    if inputs.pprof_outputs.flamegraph_available && inputs.pprof_outputs.stack_summary_available {
        report.push_str(
            "The flamegraph and stack summary aggregate sampled CPU stacks across all profiled runs. Use the flamegraph to find inclusive hotspots visually and the stack summary text file to inspect sampled call stacks directly.\n\n",
        );
        report.push_str(
            "Because pprof uses sampling, very short runs may produce sparse data. Increase `--repeats` or the synthetic case size if the flamegraph is too thin to be useful.\n\n",
        );
        report.push_str(&format!(
            "- Flamegraph: `{}`\n",
            inputs.flamegraph_path.display()
        ));
        report.push_str(&format!(
            "- Stack summary: `{}`\n",
            inputs.stack_report_path.display()
        ));
    } else if let Some(note) = inputs.pprof_outputs.note {
        report.push_str(note);
        report.push_str("\n\n");
    }
    report.push_str("\n");
    report.push_str("## Run Times\n\n");
    report.push_str("| Run | Seconds |\n");
    report.push_str("|---|---:|\n");
    for (index, elapsed) in inputs.elapsed_runs.iter().enumerate() {
        report.push_str(&format!(
            "| {} | {:.3} |\n",
            index + 1,
            elapsed.as_secs_f64()
        ));
    }
    report.push_str("\n");
    report.push_str("## Hot Functions\n\n");
    if inputs.records.is_empty() {
        report.push_str(
            "No in-process profiling records were captured. Ensure the profiling feature is enabled and the workload exercises the instrumented paths.\n\n",
        );
    } else {
        report.push_str("| Rank | Function | Calls | Total ms | Avg ms | Max ms | Profiled % |\n");
        report.push_str("|---:|---|---:|---:|---:|---:|---:|\n");
        for (index, record) in inputs.records.iter().enumerate() {
            let pct = if total_profiled.is_zero() {
                0.0
            } else {
                100.0 * record.total.as_secs_f64() / total_profiled.as_secs_f64()
            };
            report.push_str(&format!(
                "| {} | `{}` | {} | {:.3} | {:.3} | {:.3} | {:.2} |\n",
                index + 1,
                record.name,
                record.calls,
                record.total.as_secs_f64() * 1_000.0,
                record.average().as_secs_f64() * 1_000.0,
                record.max.as_secs_f64() * 1_000.0,
                pct,
            ));
        }
        report.push_str("\n");
    }

    fs::write(report_path, report)?;
    Ok(())
}
