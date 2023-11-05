use criterion::*;
use interpn::multilinear_rectilinear;
use interpn::multilinear_regular;
use interpn::utils::*;
use randn::*;

fn bench_interp(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_interp");
    for size in [100, 10_000, 250_000, 500_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("hypercube regular 2d max 2d", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let m: usize = (size as f64).sqrt() as usize;
                let nx = m / 2;
                let ny = m * 2;
                let n = nx * ny;

                let x = linspace(0.0, 100.0, nx);
                let y = linspace(0.0, 100.0, ny);
                let z = randn::<f64>(&mut rng, n);
                let mut out = vec![0.0; n];

                let xy: Vec<f64> = meshgrid(Vec::from([&x, &y]))
                    .iter()
                    .flatten()
                    .map(|xx| *xx)
                    .collect();

                b.iter(|| {
                    black_box({
                        // Initializing the interpolator local to where it will be used
                        // is about a 2x speedup
                        let dims = [nx, ny];
                        let starts = [x[0], y[0]];
                        let steps = [x[1] - x[0], y[1] - y[0]];
                        let interpolator: multilinear_regular::RegularGridInterpolator<'_, _, 2> =
                            multilinear_regular::RegularGridInterpolator::new(
                                &z[..],
                                &dims[..],
                                &starts[..],
                                &steps[..],
                            );
                        interpolator.interp(&xy[..], &mut out)
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hypercube regular interpn 2d max 10d", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let m: usize = (size as f64).sqrt() as usize;
                let nx = m / 2;
                let ny = m * 2;
                let n = nx * ny;

                let x = linspace(0.0, 100.0, nx);
                let y = linspace(0.0, 100.0, ny);
                let z = randn::<f64>(&mut rng, n);
                let mut out = vec![0.0; n];

                let xy: Vec<f64> = meshgrid(Vec::from([&x, &y]))
                    .iter()
                    .flatten()
                    .map(|xx| *xx)
                    .collect();

                b.iter(|| {
                    black_box({
                        let dims = [nx, ny];
                        let starts = [x[0], y[0]];
                        let steps = [x[1] - x[0], y[1] - y[0]];
                        multilinear_regular::interpn(
                            &xy[..],
                            &mut out,
                            &z[..],
                            &dims[..],
                            &starts[..],
                            &steps[..],
                        )
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hypercube rectilinear interpn 2d max 10d", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let m: usize = (size as f64).sqrt() as usize;
                let nx = m / 2;
                let ny = m * 2;
                let n = nx * ny;

                let x = linspace(0.0, 100.0, nx);
                let y = linspace(0.0, 100.0, ny);
                let z = randn::<f64>(&mut rng, n);
                let mut out = vec![0.0; n];

                let xy: Vec<f64> = meshgrid(Vec::from([&x, &y]))
                    .iter()
                    .flatten()
                    .map(|xx| *xx)
                    .collect();

                b.iter(|| {
                    black_box({
                        multilinear_rectilinear::interpn(&xy[..], &mut out, &z[..], &[&x, &y])
                    })
                });
            },
        );

        // group.bench_with_input(BenchmarkId::new("spliny_linear", size), size, |b, &size| {
        //     let mut rng = rng_fixed_seed();
        //     let x = randn::<f64>(&mut rng, size);
        //     let y = randn::<f64>(&mut rng, size);

        //     b.iter(|| {
        //         black_box(
        //             1,
        //             // x.iter()
        //             //     .zip(x.iter())
        //             //     .map(|(xx, yy)| xx * yy)
        //             //     .collect::<Vec<_>>(),
        //         )
        //     });
        // });
    }
    group.finish();
}

criterion_group!(benches_interp, bench_interp);
criterion_main!(benches_interp,);

mod randn {
    use rand::distributions::{Distribution, Standard};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    /// Fixed random seed to support repeatable testing
    const SEED: [u8; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
        6, 5, 4, 3, 2, 1,
    ];

    /// Get a random number generator with a const seed for repeatable testing
    pub fn rng_fixed_seed() -> StdRng {
        StdRng::from_seed(SEED)
    }

    /// Generate `n` random numbers using provided generator
    pub fn randn<T>(rng: &mut StdRng, n: usize) -> Vec<T>
    where
        Standard: Distribution<T>,
    {
        let out: Vec<T> = (0..n).map(|_| rng.gen::<T>()).collect();
        out
    }
}
