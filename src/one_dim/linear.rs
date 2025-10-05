//! Special case of 1D linear interpolation, which can be
//! significantly faster than more general N-D interpolation.

use num_traits::Float;

use super::{Extrap, Grid1D, GridSample, Interp1D};

/// Simple linear interpolation / extrapolation.
pub struct Linear1D<G> {
    grid: G,
}

impl<G> Linear1D<G> {
    pub fn new(grid: G) -> Self {
        Self { grid }
    }
}

impl<'a, T, G> Interp1D<'a, T, G> for Linear1D<G>
where
    T: Float,
    G: Grid1D<'a, T>,
{
    #[inline]
    fn eval_one(&self, loc: T) -> Result<T, &'static str> {
        let GridSample { x0, y0, x1, y1, .. } = self.grid.at(loc)?;

        let slope = (y1 - y0) / (x1 - x0);
        let dx = loc - x0;
        let v = y0 + slope * dx;

        Ok(v)
    }
}

/// Linear interpolation / extrapolation with hold-last extrapolation;
/// holds the leftmost value when extrapolating low, and the rightmost
/// value when extrapolating high.
pub struct LinearHoldLast1D<G> {
    grid: G,
}

impl<G> LinearHoldLast1D<G> {
    pub fn new(grid: G) -> Self {
        Self { grid }
    }
}

impl<'a, T, G> Interp1D<'a, T, G> for LinearHoldLast1D<G>
where
    T: Float,
    G: Grid1D<'a, T>,
{
    #[inline]
    fn eval_one(&self, loc: T) -> Result<T, &'static str> {
        let GridSample {
            x0,
            y0,
            x1,
            y1,
            extrap,
        } = self.grid.at(loc)?;

        let v = match extrap {
            Extrap::Inside => {
                let slope = (y1 - y0) / (x1 - x0);
                let dx = loc - x0;
                y0 + slope * dx
            }
            Extrap::OutsideLow => y0,
            Extrap::OutsideHigh => y1,
        };

        Ok(v)
    }
}

#[cfg(test)]
mod test {
    use crate::one_dim::{Interp1D, RectilinearGrid1D, RegularGrid1D};
    use crate::testing::{randn, rng_fixed_seed};
    use crate::utils::linspace;

    use super::{Linear1D, LinearHoldLast1D};

    #[test]
    fn test_linear_1d() {
        let rng = &mut rng_fixed_seed();

        let n = 77;

        let vals = &randn::<f64>(rng, n)[..];

        // Regular grid
        let (start, stop) = (-3.14, 314.0);
        let x_reg = linspace(start, stop, n);
        let g_reg = RegularGrid1D::new(x_reg[0], x_reg[1] - x_reg[0], vals).unwrap();

        // Rectilinear grid
        let mut x_rect = randn::<f64>(rng, n);
        x_rect.sort_unstable_by(|x, y| x.total_cmp(y));
        x_rect
            .iter_mut()
            .for_each(|x| *x = (*x * (stop - start)) + start);
        let g_rect = RectilinearGrid1D::new(&x_rect, vals).unwrap();

        // Interpolators
        let lin_reg = Linear1D::new(g_reg);
        let lin_rect = Linear1D::new(g_rect);

        let linhl_reg = LinearHoldLast1D::new(g_reg);
        let linhl_rect = LinearHoldLast1D::new(g_rect);

        // Observations under both interpolation and extrapolation
        let mut locs = randn::<f64>(rng, 3 * n);
        locs.iter_mut()
            .for_each(|x| *x = (*x * 2.0 * (stop - start)) + 2.0 * start);

        let y_lin_reg = lin_reg.eval_alloc(&locs).unwrap();
        let y_lin_rect = lin_rect.eval_alloc(&locs).unwrap();
        let y_linhl_reg = linhl_reg.eval_alloc(&locs).unwrap();
        let y_linhl_rect = linhl_rect.eval_alloc(&locs).unwrap();

        // Check
        for (xs, ys, hold) in [
            (&x_reg, y_lin_reg, false),
            (&x_rect, y_lin_rect, false),
            (&x_reg, y_linhl_reg, true),
            (&x_rect, y_linhl_rect, true),
        ] {
            for i in 0..locs.len() {
                let loc = locs[i];
                let y = ys[i];
                let j: usize = ((xs.partition_point(|v| v < &loc) as isize - 1).max(0) as usize)
                    .min(xs.len() - 2);

                let xleft = xs[j];
                let xright = xs[j + 1];
                let yleft = vals[j];
                let yright = vals[j + 1];

                let slope = (yright - yleft) / (xright - xleft);
                let dx = loc - xleft;

                let ymax = yleft.max(yright);
                let ymin = yleft.min(yright);

                // Interpolation
                if loc >= xs[0] && loc <= xs[n - 1] {
                    assert!(y <= ymax && y >= ymin);
                    assert!(
                        loc >= xleft && loc <= xright,
                        "Didn't find the correct cell"
                    );
                } else if loc > xs[n - 1] && hold {
                    let y_expected = vals[n - 1];
                    assert!(((y - y_expected) / y_expected).abs() < 1e-12);
                    continue;
                } else if loc < xs[0] && hold {
                    let y_expected = vals[0];
                    assert!(((y - y_expected) / y_expected).abs() < 1e-12);
                    continue;
                }

                let y_expected = yleft + slope * dx;
                assert!(((y - y_expected) / y_expected).abs() < 1e-12);
            }
        }
    }
}
