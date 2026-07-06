use std::{
    error::Error,
    fs,
    hint::black_box,
    io,
    path::Path,
    time::{Duration, Instant},
};

use interpn::{GridInterpMethod, GridKind, interpn_serial};
use ninterp::{
    ndarray::prelude::*,
    prelude::{Extrapolate, Interp1D, Interp2D, Interp3D, Interpolator, strategy},
};
use plotly::{
    Layout, Plot, Scatter,
    common::{Line, Mode, Title},
    layout::{Axis, AxisType},
};

const DIMENSIONS: &[usize] = &[1, 2, 3];
const SAMPLE_COUNTS: &[usize] = &[1, 4_096];
const GRID_SPECS: &[(f64, f64, usize, f64)] = &[
    (-1.0, 1.0, 13, 1.35),
    (-0.75, 0.95, 11, 1.2),
    (-0.5, 0.8, 9, 1.1),
];
const MIN_REPEATS: usize = 5;
const MAX_REPEATS: usize = 2_000;
const TARGET_TIMING: Duration = Duration::from_millis(250);

fn main() -> Result<(), Box<dyn Error>> {
    let mut results = Vec::with_capacity(DIMENSIONS.len() * SAMPLE_COUNTS.len());
    for &sample_count in SAMPLE_COUNTS {
        for &dimension in DIMENSIONS {
            results.push(bench_case(dimension, sample_count)?);
        }
    }

    let mut plot = Plot::new();
    for &sample_count in SAMPLE_COUNTS {
        let sample_results = results
            .iter()
            .filter(|result| result.sample_count == sample_count)
            .collect::<Vec<_>>();
        plot.add_trace(
            Scatter::new(
                sample_results
                    .iter()
                    .map(|result| result.dimension as f64)
                    .collect::<Vec<_>>(),
                sample_results
                    .iter()
                    .map(|result| result.throughput_ratio())
                    .collect::<Vec<_>>(),
            )
            .mode(Mode::Lines)
            .name(format!(
                "{sample_count} sample{}",
                if sample_count == 1 { "" } else { "s" }
            ))
            .line(Line::new().width(3.0)),
        );
    }
    plot.set_layout(
        Layout::new()
            .title(Title::with_text(
                "Rectilinear linear interpolation throughput ratio",
            ))
            .x_axis(
                Axis::new()
                    .title(Title::with_text("dimensions"))
                    .type_(AxisType::Linear),
            )
            .y_axis(
                Axis::new()
                    .title(Title::with_text("interpn throughput / ninterp throughput"))
                    .type_(AxisType::Linear),
            ),
    );

    let output = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("ninterp_timing.html");
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    plot.write_html(&output);

    println!("wrote {}", output.display());
    println!(
        "{:>4} {:>8} {:>14} {:>14} {:>14} {:>14} {:>10} {:>12} {:>12} {:>14}",
        "dims",
        "samples",
        "interpn ms",
        "ninterp ms",
        "interpn M/s",
        "ninterp M/s",
        "ratio",
        "interpn reps",
        "ninterp reps",
        "max delta",
    );
    for result in &results {
        println!(
            "{:>4} {:>8} {:>14.6} {:>14.6} {:>14.3} {:>14.3} {:>10.3} {:>12} {:>12} {:>14.3e}",
            result.dimension,
            result.sample_count,
            result.interpn_avg.as_secs_f64() * 1_000.0,
            result.ninterp_avg.as_secs_f64() * 1_000.0,
            result.throughput_mps(result.interpn_avg),
            result.throughput_mps(result.ninterp_avg),
            result.throughput_ratio(),
            result.interpn_repeats,
            result.ninterp_repeats,
            result.max_pairwise_delta,
        );
    }

    Ok(())
}

fn bench_case(dimension: usize, sample_count: usize) -> Result<TimingResult, Box<dyn Error>> {
    let grids = build_grids(dimension);
    let values = build_values(&grids);
    let observations = build_observations(&grids, sample_count);
    let grid_slices = grids.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let observation_slices = observations.iter().map(Vec::as_slice).collect::<Vec<_>>();

    match dimension {
        1 => measure_case(
            dimension,
            sample_count,
            &grid_slices,
            &values,
            &observation_slices,
            |out| run_ninterp_1d(&grids, &values, &observations, out),
        ),
        2 => measure_case(
            dimension,
            sample_count,
            &grid_slices,
            &values,
            &observation_slices,
            |out| run_ninterp_2d(&grids, &values, &observations, out),
        ),
        3 => measure_case(
            dimension,
            sample_count,
            &grid_slices,
            &values,
            &observation_slices,
            |out| run_ninterp_3d(&grids, &values, &observations, out),
        ),
        _ => Err(io_error("only 1D, 2D, and 3D are supported").into()),
    }
}

fn measure_case<N>(
    dimension: usize,
    sample_count: usize,
    grid_slices: &[&[f64]],
    values: &[f64],
    observation_slices: &[&[f64]],
    mut run_ninterp: N,
) -> Result<TimingResult, Box<dyn Error>>
where
    N: FnMut(&mut [f64]) -> Result<(), Box<dyn Error>>,
{
    let mut interpn_values = vec![0.0; sample_count];
    let mut ninterp_values = vec![0.0; sample_count];

    run_interpn(grid_slices, values, observation_slices, &mut interpn_values)?;
    run_ninterp(&mut ninterp_values)?;

    let max_pairwise_delta = interpn_values
        .iter()
        .zip(ninterp_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let interpn_repeats = choose_repeats(|| {
        run_interpn(grid_slices, values, observation_slices, &mut interpn_values)
    })?;
    let ninterp_repeats = choose_repeats(|| run_ninterp(&mut ninterp_values))?;

    let interpn_avg = time_repeated(interpn_repeats, || {
        run_interpn(grid_slices, values, observation_slices, &mut interpn_values)
    })?;
    let ninterp_avg = time_repeated(ninterp_repeats, || run_ninterp(&mut ninterp_values))?;

    Ok(TimingResult {
        dimension,
        sample_count,
        interpn_avg,
        ninterp_avg,
        interpn_repeats,
        ninterp_repeats,
        max_pairwise_delta,
    })
}

#[derive(Clone, Copy, Debug)]
struct TimingResult {
    dimension: usize,
    sample_count: usize,
    interpn_avg: Duration,
    ninterp_avg: Duration,
    interpn_repeats: usize,
    ninterp_repeats: usize,
    max_pairwise_delta: f64,
}

impl TimingResult {
    fn throughput_mps(self, elapsed: Duration) -> f64 {
        self.sample_count as f64 / elapsed.as_secs_f64() / 1.0e6
    }

    fn throughput_ratio(self) -> f64 {
        self.ninterp_avg.as_secs_f64() / self.interpn_avg.as_secs_f64()
    }
}

fn run_interpn(
    grids: &[&[f64]],
    values: &[f64],
    observations: &[&[f64]],
    out: &mut [f64],
) -> Result<(), io::Error> {
    interpn_serial(
        grids,
        values,
        observations,
        out,
        GridInterpMethod::Linear,
        Some(GridKind::Rectilinear),
        true,
        None,
    )
    .map_err(io_error)?;
    black_box(&out);
    Ok(())
}

fn run_ninterp_1d(
    grids: &[Vec<f64>],
    values: &[f64],
    observations: &[Vec<f64>],
    out: &mut [f64],
) -> Result<(), Box<dyn Error>> {
    let interp = Interp1D::new(
        Array1::from_vec(grids[0].clone()),
        Array1::from_vec(values.to_vec()),
        strategy::Linear,
        Extrapolate::Error,
    )?;
    for (i, out) in out.iter_mut().enumerate() {
        *out = interp.interpolate(&[observations[0][i]])?;
    }
    black_box(&out);
    Ok(())
}

fn run_ninterp_2d(
    grids: &[Vec<f64>],
    values: &[f64],
    observations: &[Vec<f64>],
    out: &mut [f64],
) -> Result<(), Box<dyn Error>> {
    let shape = (grids[0].len(), grids[1].len());
    let interp = Interp2D::new(
        Array1::from_vec(grids[0].clone()),
        Array1::from_vec(grids[1].clone()),
        Array2::from_shape_vec(shape, values.to_vec())?,
        strategy::Linear,
        Extrapolate::Error,
    )?;
    for (i, out) in out.iter_mut().enumerate() {
        *out = interp.interpolate(&[observations[0][i], observations[1][i]])?;
    }
    black_box(&out);
    Ok(())
}

fn run_ninterp_3d(
    grids: &[Vec<f64>],
    values: &[f64],
    observations: &[Vec<f64>],
    out: &mut [f64],
) -> Result<(), Box<dyn Error>> {
    let shape = (grids[0].len(), grids[1].len(), grids[2].len());
    let interp = Interp3D::new(
        Array1::from_vec(grids[0].clone()),
        Array1::from_vec(grids[1].clone()),
        Array1::from_vec(grids[2].clone()),
        Array3::from_shape_vec(shape, values.to_vec())?,
        strategy::Linear,
        Extrapolate::Error,
    )?;
    for (i, out) in out.iter_mut().enumerate() {
        *out = interp.interpolate(&[observations[0][i], observations[1][i], observations[2][i]])?;
    }
    black_box(&out);
    Ok(())
}

fn choose_repeats<F, E>(mut run_once: F) -> Result<usize, E>
where
    F: FnMut() -> Result<(), E>,
{
    let start = Instant::now();
    run_once()?;
    let elapsed = start.elapsed();
    let repeats = if elapsed.is_zero() {
        MAX_REPEATS
    } else {
        (TARGET_TIMING.as_secs_f64() / elapsed.as_secs_f64()).ceil() as usize
    };
    Ok(repeats.clamp(MIN_REPEATS, MAX_REPEATS))
}

fn time_repeated<F, E>(repeats: usize, mut run_once: F) -> Result<Duration, E>
where
    F: FnMut() -> Result<(), E>,
{
    let start = Instant::now();
    for _ in 0..repeats {
        run_once()?;
    }
    Ok(start.elapsed() / repeats as u32)
}

fn build_grids(dimension: usize) -> Vec<Vec<f64>> {
    GRID_SPECS
        .iter()
        .take(dimension)
        .map(|&(start, stop, n, exponent)| rectilinear_grid(start, stop, n, exponent))
        .collect()
}

fn build_observations(grids: &[Vec<f64>], sample_count: usize) -> Vec<Vec<f64>> {
    grids
        .iter()
        .enumerate()
        .map(|(dimension, grid)| {
            let start = grid[0];
            let stop = *grid.last().unwrap();
            if sample_count == 1 {
                vec![(start + stop) * 0.5]
            } else {
                let exponent = 1.0 + dimension as f64 * 0.15;
                (0..sample_count)
                    .map(|i| {
                        let t = i as f64 / (sample_count - 1) as f64;
                        start + (stop - start) * t.powf(exponent)
                    })
                    .collect()
            }
        })
        .collect()
}

fn build_values(grids: &[Vec<f64>]) -> Vec<f64> {
    match grids.len() {
        1 => grids[0].iter().map(|&x| truth(&[x])).collect(),
        2 => {
            let mut values = Vec::with_capacity(grids[0].len() * grids[1].len());
            for &x in &grids[0] {
                for &y in &grids[1] {
                    values.push(truth(&[x, y]));
                }
            }
            values
        }
        3 => {
            let mut values = Vec::with_capacity(grids[0].len() * grids[1].len() * grids[2].len());
            for &x in &grids[0] {
                for &y in &grids[1] {
                    for &z in &grids[2] {
                        values.push(truth(&[x, y, z]));
                    }
                }
            }
            values
        }
        _ => panic!("only 1D, 2D, and 3D are supported"),
    }
}

fn truth(x: &[f64]) -> f64 {
    let x0 = x[0];
    let x1 = x.get(1).copied().unwrap_or(0.37);
    let x2 = x.get(2).copied().unwrap_or(-0.2);
    (std::f64::consts::PI * x0).sin() * (1.3 * x1).cos()
        + 0.2 * x0 * x1
        + 0.15 * (2.0 * x2).sin()
        + 0.05 * x0 * x2
}

fn rectilinear_grid(start: f64, stop: f64, n: usize, exponent: f64) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                start + (stop - start) * t.powf(exponent)
            })
            .collect(),
    }
}

fn io_error(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::Other, message)
}
