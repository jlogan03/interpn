//! Building this module successfully guarantees that the library is no-std compatible

#![no_std]
#![no_main]

use core::panic::PanicInfo;

use interpn::multilinear::{regular, rectilinear};
use interpn::multicubic;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // We can't print, so there's not much to do here
    loop {}
}

#[no_mangle]
pub fn _start() -> ! {
    // Define a grid
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let y = [0.0_f64, 1.0, 2.0, 3.0];
    
    // Grid input for rectilinear method
    let grids = &[&x[..], &y[..]];
    
    // Grid input for regular grid method
    let dims = [x.len(), y.len()];
    let starts = [x[0], y[0]];
    let steps = [x[1] - x[0], y[1] - y[0]];
    
    // Values at grid points
    let z = [2.0; 16];
    
    // Observation points to interpolate/extrapolate
    let xobs = [0.0_f64, 5.0];
    let yobs = [-1.0, 3.0];
    let obs = [&xobs[..], &yobs[..]];
    
    // Storage for output
    let mut out = [0.0; 2];

    regular::interpn(&dims, &starts, &steps, &z, &obs, &mut out).unwrap();
    rectilinear::interpn(grids, &z, &obs, &mut out).unwrap();
    multicubic::regular::interpn(&dims, &starts, &steps, &z, false, &obs, &mut out).unwrap();
    multicubic::rectilinear::interpn(grids, &z, false, &obs, &mut out).unwrap();

    loop {} // We don't actually run this, just compile it
}
