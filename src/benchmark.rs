use std::ffi::{c_char, CString};

use super::Image;


#[repr(C)]
pub struct GPUStats {
    pub host_to_device_time_cuda: f32,
    pub kernel_execution_time_cuda: f32,
    pub device_to_host_time_cuda: f32,

    pub host_to_device_time_chrono: f32,
    pub kernel_execution_time_chrono: f32,
    pub device_to_host_time_chrono: f32,
}

extern "C" {
    // // Function declarations for the C++ functions
    fn load_image(file_path: *const c_char) -> Image;
    fn save_image(file_path: *const c_char, image: *const Image);
    #[allow(dead_code)]
    fn free_image(image: Image);

    fn convert_to_grayscale_cpu(image: *mut Image);
    fn convert_to_grayscale_gpu(image: *mut Image) -> GPUStats;

    #[allow(dead_code)]
    fn say_hello();
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

pub fn grayscale_cpu(image: &mut Image) {
    unsafe {
        convert_to_grayscale_cpu(image);
    }
}

pub fn grayscale_gpu(image: &mut Image)  -> GPUStats {
    unsafe {
        convert_to_grayscale_gpu(image)
    }
}
