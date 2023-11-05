//! Building this module successfully guarantees that the library is no-std compatible

#![no_std]
#![no_main]

use core::panic::PanicInfo;

use interpn::*;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // We can't print, so there's not much to do here
    loop {}
}

#[no_mangle]
pub fn _start() -> ! {
    // let a = [1.0_f64, 2.0, 3.0, 4.0];
    // let b = [1.0_f64, 2.0, 3.0, 4.0];
    // let mut c = [0.0; 4];

    // add(&mut array(&a), &mut array(&b)).eval_into_slice(&mut c);

    loop {} // We don't actually run this, just compile it
}
