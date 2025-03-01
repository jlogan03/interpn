//! Piecewise-constant 1D interpolators

use num_traits::Float;

use super::{Extrap, Grid1D, GridSample, Interp1D};

/// Hold-last piecewise constant interpolation
pub struct Left1D<G> {
    grid: G,
}

impl<G> Left1D<G> {
    pub fn new(grid: G) -> Self {
        Self { grid }
    }
}

impl<'a, T, G> Interp1D<'a, T, G> for Left1D<G>
where
    T: Float,
    G: Grid1D<'a, T>,
{
    #[inline]
    fn eval_one(&self, loc: T) -> Result<T, &'static str> {
        let GridSample {
            x0: _,
            y0,
            x1: _,
            y1,
            extrap,
        } = self.grid.at(loc)?;

        let v = match extrap {
            Extrap::OutsideHigh => y1,
            _ => y0,
        };

        Ok(v)
    }
}

/// Hold-next piecewise-constant interpolation
pub struct Right1D<G> {
    grid: G,
}

impl<G> Right1D<G> {
    pub fn new(grid: G) -> Self {
        Self { grid }
    }
}

impl<'a, T, G> Interp1D<'a, T, G> for Right1D<G>
where
    T: Float,
    G: Grid1D<'a, T>,
{
    #[inline]
    fn eval_one(&self, loc: T) -> Result<T, &'static str> {
        let GridSample {
            x0: _,
            y0,
            x1: _,
            y1,
            extrap,
        } = self.grid.at(loc)?;

        let v = match extrap {
            Extrap::OutsideLow => y0,
            _ => y1,
        };

        Ok(v)
    }
}

/// Nearest-value piecewise-constant interpolation.
/// In the event of a tie, the left value is taken.
pub struct Nearest1D<G> {
    grid: G,
}

impl<G> Nearest1D<G> {
    pub fn new(grid: G) -> Self {
        Self { grid }
    }
}

impl<'a, T, G> Interp1D<'a, T, G> for Nearest1D<G>
where
    T: Float,
    G: Grid1D<'a, T>,
{
    #[inline]
    fn eval_one(&self, loc: T) -> Result<T, &'static str> {
        let GridSample { x0, y0, x1, y1, .. } = self.grid.at(loc)?;

        let dx0 = (loc - x0).abs();
        let dx1 = (loc - x1).abs();

        let v = match dx1 >= dx0 {
            true => y0,
            false => y1,
        };

        Ok(v)
    }
}

#[cfg(test)]
mod test {
    use crate::one_dim::{Interp1D, RegularGrid1D};
    use crate::testing::{randn, rng_fixed_seed};
    use crate::utils::linspace;

    use super::{Left1D, Nearest1D, Right1D};

    #[test]
    fn test_hold_1d() {
        let rng = &mut rng_fixed_seed();

        let n = 77;

        let vals = &randn::<f64>(rng, n)[..];

        // Regular grid
        let (start, stop) = (-3.14, 314.0);
        let x_reg = linspace(start, stop, n);
        let g_reg = RegularGrid1D::new(x_reg[0], x_reg[1] - x_reg[0], vals).unwrap();

        // Interpolators
        let left_reg = Left1D::new(g_reg);
        let right_reg = Right1D::new(g_reg);
        let nearest_reg = Nearest1D::new(g_reg);

        // Observations under both interpolation and extrapolation
        let mut locs = randn::<f64>(rng, 3 * n);
        locs.iter_mut()
            .for_each(|x| *x = (*x * 2.0 * (stop - start)) + 2.0 * start);

        let y_lreg = left_reg.eval_alloc(&locs).unwrap();
        let y_rreg = right_reg.eval_alloc(&locs).unwrap();
        let y_nreg = nearest_reg.eval_alloc(&locs).unwrap();

        // Check
        for i in 0..locs.len() {
            let loc = locs[i];
            let j: usize = ((x_reg.partition_point(|v| v < &loc) as isize - 1).max(0) as usize)
                .min(x_reg.len() - 2);

            let xleft = x_reg[j];
            let xright = x_reg[j + 1];
            let yleft = vals[j];
            let yright = vals[j + 1];

            // Interpolation
            if loc >= x_reg[0] && loc <= x_reg[n - 1] {
                assert!(
                    loc >= xleft && loc <= xright,
                    "Didn't find the correct cell"
                );

                assert_eq!(y_lreg[i], yleft);
                assert_eq!(y_rreg[i], yright);
            } else if loc > x_reg[n - 1] {
                assert_eq!(y_lreg[i], yright);
                assert_eq!(y_rreg[i], yright);
            } else if loc < x_reg[0] {
                assert_eq!(y_lreg[i], yleft);
                assert_eq!(y_rreg[i], yleft);
            }

            let y_nearest = match (loc - xleft) <= (xright - loc) {
                true => yleft,
                false => yright,
            };
            assert_eq!(y_nreg[i], y_nearest);
        }
    }
}
