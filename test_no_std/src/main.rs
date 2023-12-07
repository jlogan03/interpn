//! Building this module successfully guarantees that the library is no-std compatible

#![no_std]
#![no_main]

use core::panic::PanicInfo;

use interpn::multilinear::{regular, rectilinear};

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // We can't print, so there's not much to do here
    loop {}
}

#[no_mangle]
pub fn _start() -> ! {
    let x = [1.0_f64, 2.0];
    let y = [1.0_f64, 2.0];
    let grids = &[&x[..], &y[..]];

    let z = [2.0; 4];

    let mut out = [0.0; 1];

    let dims = [x.len(), y.len()];
    let starts = [x[0], y[0]];
    let steps = [1.0, 1.0];

    let obs = [&[0.0_f64][..], &[0.0_f64][..]]; // Slightly weird syntax to get slice of slice without vec

    regular::interpn(&dims, &starts, &steps, &z, &obs, &mut out).unwrap();
    rectilinear::interpn(grids, &z, &obs, &mut out).unwrap();

    loop {} // We don't actually run this, just compile it
}
