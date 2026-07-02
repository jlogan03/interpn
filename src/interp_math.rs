use num_traits::Float;

use crate::mul_add;

#[inline]
pub(crate) fn dot4<T: Float>(weights: [T; 4], vals: [T; 4]) -> T {
    let lo = mul_add(weights[1], vals[1], weights[0] * vals[0]);
    let hi = mul_add(weights[3], vals[3], weights[2] * vals[2]);
    lo + hi
}

#[inline]
pub(crate) fn hermite_basis<T: Float>(t: T) -> [T; 4] {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let t2 = t * t;
    let t3 = t2 * t;

    [
        mul_add(two, t3, mul_add(-three, t2, one)),
        mul_add(t, mul_add(t, t - two, one), T::zero()),
        mul_add(-two, t3, three * t2),
        mul_add(t2, t - one, T::zero()),
    ]
}

#[inline]
pub(crate) fn hermite_basis_derivative<T: Float>(t: T) -> [T; 4] {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let six = three + three;
    let t2 = t * t;

    [
        six * (t2 - t),
        three * t2 - (two + two) * t + one,
        six * (t - t2),
        three * t2 - two * t,
    ]
}
