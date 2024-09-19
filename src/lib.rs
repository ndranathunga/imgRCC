// #[cfg(feature = "benchmark")]
pub mod benchmark;

use std::ffi::{c_char, CString};
use std::os::raw::{c_int, c_uchar};

// Define Image struct in Rust matching C++ Image structure
#[repr(C)]
pub struct Image {
    width: c_int,
    height: c_int,
    channels: c_int,
    data: *mut c_uchar,
}

extern "C" {
    // // Function declarations for the C++ functions
    fn load_image(file_path: *const c_char) -> Image;
    fn save_image(file_path: *const c_char, image: *const Image);
    fn free_image(image: Image);

    fn convert_to_grayscale_cpu(image: *mut Image);
    fn convert_to_grayscale_gpu(image: *mut Image);

    fn say_hello();
}

// Helper functions in Rust to work with the Image

pub fn grayscale_cpu(image: &mut Image) {
    unsafe {
        convert_to_grayscale_cpu(image);
    }
}

pub fn grayscale_gpu(image: &mut Image) {
    unsafe {
        convert_to_grayscale_gpu(image);
    }
}

pub fn load_image_(file_path: &str) -> Image {
    let c_file_path = CString::new(file_path).expect("CString::new failed");
    unsafe { load_image(c_file_path.as_ptr()) }
}

pub fn save_image_(file_path: &str, image: &Image) {
    let c_file_path = CString::new(file_path).expect("CString::new failed");
    unsafe {
        save_image(c_file_path.as_ptr(), image);
    }
}

pub fn free_image_(image: Image) {
    unsafe {
        free_image(image);
    }
}

// impl Drop for Image {
//     fn drop(&mut self) {
//         unsafe {
//             free_image(*self);
//         }
//     }
// }

pub fn say_hello_() {
    unsafe {
        say_hello();
    }
}
