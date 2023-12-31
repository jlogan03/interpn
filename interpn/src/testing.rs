use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Fixed random seed to support repeatable testing
const SEED: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
    5, 4, 3, 2, 1,
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
