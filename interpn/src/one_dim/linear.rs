//! Special case of 1D linear interpolation

use num_traits::Float;

use super::{Extrap, Grid1D, Interp1D};

/// Simple linear interpolation / extrapolation.
pub struct Linear1D<G> {
    grid: G
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
        let ((x0, y0), (x1, y1), _extrap) = self.grid.at(loc)?;

        let slope = (y1 - y0) / (x1 - x0);
        let dx = loc - x0;
        let v = x0 + slope * dx;

        Ok(v)
    }
}

/// Linear interpolation / extrapolation with hold-last extrapolation;
/// holds the leftmost value when extrapolating low, and the rightmost
/// value when extrapolating high.
pub struct LinearHoldLast1D<G> {
    grid: G
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
        let ((x0, y0), (x1, y1), extrap) = self.grid.at(loc)?;

        let v = match extrap {
            Extrap::Inside => {
                let slope = (y1 - y0) / (x1 - x0);
                let dx = loc - x0;
                x0 + slope * dx
            },
            Extrap::OutsideLow => {
                y0
            },
            Extrap::OutsideHigh => {
                y1
            }
        };

        Ok(v)
    }
}


#[cfg(test)]
mod test {
    #[test]
    fn test_linear_1d() {
        
    }
}