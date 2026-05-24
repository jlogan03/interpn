//! Scalar fused multiply-add helpers.
//!
//! These helpers centralize optional use of `Float::mul_add` under the `fma`
//! feature so call sites can opt into fused arithmetic without repeating
//! feature-gated branches.

use num_traits::Float;

/// Returns `mul_lhs * mul_rhs + addend`.
///
/// When the `fma` feature is enabled this uses `Float::mul_add`; otherwise it
/// falls back to the ordinary multiply-then-add expression.
#[inline]
pub(crate) fn mul_add<T>(mul_lhs: T, mul_rhs: T, addend: T) -> T
where
    T: Float,
{
    #[cfg(feature = "fma")]
    {
        mul_lhs.mul_add(mul_rhs, addend)
    }
    #[cfg(not(feature = "fma"))]
    {
        mul_lhs * mul_rhs + addend
    }
}

#[cfg(test)]
mod tests {
    use super::mul_add;

    #[test]
    fn mul_add_matches_plain_expression() {
        assert_eq!(mul_add(1.25_f64, -3.0, 0.5), 1.25 * -3.0 + 0.5);
        assert_eq!(mul_add(1.25_f32, -3.0, 0.5), 1.25 * -3.0 + 0.5);
    }
}
