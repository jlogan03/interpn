//! Special-cases for one-dimensional interpolation, which can use
//! a more lightweight evaluation pattern than the multidimensional methods,
//! and can support logical operations like hold-last that don't make sense
//! in a multidimensional context.

pub mod hold;
pub mod linear;

use num_traits::{Float, NumCast};

/// Extrapolation flag
pub enum Extrap {
    Inside,
    OutsideLow,
    OutsideHigh,
}

/// A point in a grid
pub struct GridSample<T> {
    pub x0: T,
    pub y0: T,
    pub x1: T,
    pub y1: T,
    pub extrap: Extrap,
}

/// A regular or rectilinear 1D grid
pub trait Grid1D<'a, T: Float> {
    /// Get the left and right values and their locations
    /// for an observation point at a given location
    /// like ((val0, loc0), (val1, loc1)).
    ///
    /// For observation points outside the grid, the returned
    /// locations may not bracket the observation point.
    ///
    /// It is highly recommended to inline implementations of this function.
    fn at(&self, loc: T) -> Result<GridSample<T>, &'static str>;
}

/// A one-dimensional interpolator.
pub trait Interp1D<'a, T: Float, G: Grid1D<'a, T>> {
    /// Evaluate the interpolant at an observation point.
    ///
    /// It is highly recommended to inline implementations of this function.
    fn eval_one(&self, loc: T) -> Result<T, &'static str>;

    /// Evaluate the interpolant at a set of observation points.
    ///
    /// It is highly recommended to inline implementations of this function.
    #[inline]
    fn eval(&self, locs: &[T], out: &mut [T]) -> Result<(), &'static str> {
        if locs.len() != out.len() {
            return Err("Length mismatch");
        }

        for i in 0..locs.len() {
            out[i] = self.eval_one(locs[i])?;
        }

        Ok(())
    }

    /// Evaluate the interpolant at a set of observation points, allocating
    /// for the output values for convenience.
    ///
    /// It is highly recommended to inline implementations of this function.
    #[cfg(feature = "std")]
    #[inline]
    fn eval_alloc(&self, locs: &[T]) -> Result<Vec<T>, &'static str> {
        let mut out = vec![T::zero(); locs.len()];
        self.eval(locs, &mut out)?;
        Ok(out)
    }
}

/// A regular grid, which has the same spacing between each point.
#[derive(Clone, Copy)]
pub struct RegularGrid1D<'a, T: Float> {
    start: T,
    stop: T,
    step: T,
    vals: &'a [T],
}

impl<'a, T: Float> RegularGrid1D<'a, T> {
    pub fn new(start: T, step: T, vals: &'a [T]) -> Result<Self, &'static str> {
        let stop =
            start + step * <T as NumCast>::from(vals.len() - 1).ok_or("Unrepresentable number")?;
        Ok(Self {
            start,
            stop,
            step,
            vals,
        })
    }

    /// Get the index of the lower corner of the containing grid cell
    #[inline]
    pub fn index(&self, loc: T) -> Result<(usize, Extrap), &'static str> {
        let extrap = match loc {
            x if x > self.stop => Extrap::OutsideHigh,
            x if x < self.start => Extrap::OutsideLow,
            _ => Extrap::Inside,
        };

        // Nominal location may be outside the grid
        let i = T::floor((loc - self.start) / self.step);

        // Clip to inside of grid
        let i = <isize as NumCast>::from(i)
            .ok_or("Unrepresentable number")?
            .max(0)
            .min((self.vals.len() - 2) as isize) as usize;

        Ok((i, extrap))
    }
}

impl<'a, T: Float> Grid1D<'a, T> for RegularGrid1D<'a, T> {
    #[inline]
    fn at(&self, loc: T) -> Result<GridSample<T>, &'static str> {
        let (i, extrap) = self.index(loc)?;

        let x0 =
            self.start + self.step * <T as NumCast>::from(i).ok_or("Unrepresentable number")?;
        let x1 = x0 + self.step;

        let (y0, y1) = (self.vals[i], self.vals[i + 1]);

        Ok(GridSample {
            x0,
            y0,
            x1,
            y1,
            extrap,
        })
    }
}

/// A rectilinear grid, which may have uneven spacing.
#[derive(Clone, Copy)]
pub struct RectilinearGrid1D<'a, T: Float> {
    grid: &'a [T],
    vals: &'a [T],
}

impl<'a, T: Float> RectilinearGrid1D<'a, T> {
    pub fn new(grid: &'a [T], vals: &'a [T]) -> Result<Self, &'static str> {
        if grid.len() != vals.len() || grid.len() < 2 {
            return Err("Length mismatch");
        }

        Ok(Self { grid, vals })
    }

    #[inline]
    pub fn index(&self, loc: T) -> Result<(usize, Extrap), &'static str> {
        let i = ((self.grid.partition_point(|v| v < &loc) as isize - 1).max(0) as usize)
            .min(self.grid.len() - 2);

        let extrap = match loc {
            x if x < self.grid[0] => Extrap::OutsideLow,
            x if x > self.grid[self.grid.len() - 1] => Extrap::OutsideHigh,
            _ => Extrap::Inside,
        };

        Ok((i, extrap))
    }
}

impl<'a, T: Float> Grid1D<'a, T> for RectilinearGrid1D<'a, T> {
    #[inline]
    fn at(&self, loc: T) -> Result<GridSample<T>, &'static str> {
        let (i, extrap) = self.index(loc)?;

        let (x0, x1) = (self.grid[i], self.grid[i + 1]);
        let (y0, y1) = (self.vals[i], self.vals[i + 1]);

        Ok(GridSample {
            x0,
            y0,
            x1,
            y1,
            extrap,
        })
    }
}
