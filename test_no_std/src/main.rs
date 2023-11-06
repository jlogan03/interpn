//! Building this module successfully guarantees that the library is no-std compatible

#![no_std]
#![no_main]

use core::panic::PanicInfo;

use interpn::{multilinear_rectilinear, multilinear_regular};

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // We can't print, so there's not much to do here
    loop {}
}

#[no_mangle]
pub fn _start() -> ! {
    let x = [1.0_f64, 2.0];
    let y = [1.0_f64, 2.0];

    let mut z = [2.0; 4];

    let mut out = [0.0; 4];

    let dims = [nx, ny];
    let starts = [x[0], y[0]];
    let steps = [1.0, 1.0];

    let xy = [0.0, 0.0];

    multilinear_regular::interpn(
        &xy[..],
        &mut out[..],
        &z[..],
        &dims[..],
        &starts[..],
        &steps[..],
    );

    loop {} // We don't actually run this, just compile it
}
