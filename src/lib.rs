// #[cfg(feature = "benchmark")] // currently removed because no need for specific benchmarking codes
// pub mod benchmark;

use std::ffi::{c_char, CString};
use std::os::raw::{c_int, c_uchar};

// Define Image struct in Rust matching C++ Image structure
#[repr(C)]
pub struct Image {
    width: c_int,
    height: c_int,
    channels: c_int,
    pub data: *mut c_uchar,
    device: Device,
}

#[derive(PartialEq)]
#[repr(C)]
pub enum Device {
    CPU,
    GPU,
}

extern "C" {
    fn load_image_cpu(file_path: *const c_char) -> Image;
    fn save_image_cpu(file_path: *const c_char, image: *mut Image);
    fn free_image_cpu(image: *mut Image);

    fn load_image_gpu(file_path: *const c_char) -> Image;
    fn save_image_gpu(file_path: *const c_char, image: *mut Image);
    fn free_image_gpu(image: *mut Image);

    fn convert_to_grayscale_cpu(image: *mut Image);
    fn convert_to_grayscale_gpu(image: *mut Image);

    fn transfer_to_gpu(image: *mut Image);
    fn transfer_to_cpu(image: *mut Image);
}

// Public API

impl Image {
    /// Load an image from the specified file path.
    /// The image is loaded to the CPU by default.
    /// The image is returned as an Image struct.
    pub fn load(file_path: &str) -> Image {
        let c_file_path = CString::new(file_path).expect("CString::new failed");
        let mut image = unsafe { load_image_cpu(c_file_path.as_ptr()) };
        image.device = Device::CPU;
        image
    }

    /// Load an image from the specified file path to specified device (CPU or GPU).
    /// The image is returned as an Image struct.
    /// The image is loaded to the CPU by default.
    pub fn load_to_device(file_path: &str, device: Device) -> Image {
        let c_file_path = CString::new(file_path).expect("CString::new failed");
        let mut image = match device {
            Device::CPU => unsafe { load_image_cpu(c_file_path.as_ptr()) },
            Device::GPU => unsafe { load_image_gpu(c_file_path.as_ptr()) },
        };
        image.device = device;
        image
    }

    pub fn save(&self, file_path: &str) {
        let c_file_path = CString::new(file_path).expect("CString::new failed");
        match self.device {
            Device::CPU => unsafe {
                save_image_cpu(c_file_path.as_ptr(), self as *const Image as *mut Image);
            },
            Device::GPU => unsafe {
                save_image_gpu(c_file_path.as_ptr(), self as *const Image as *mut Image);
            },
        }
    }

    /// Convert the image to grayscale based on its current device.
    pub fn grayscale(&mut self) {
        match self.device {
            Device::CPU => unsafe {
                convert_to_grayscale_cpu(self);
            },
            Device::GPU => unsafe {
                convert_to_grayscale_gpu(self);
            },
        }
    }

    /// Move the image to the specified device (CPU or GPU).
    pub fn to(&mut self, device: Device) {
        match device {
            Device::CPU => {
                if self.device == Device::GPU {
                    unsafe {
                        transfer_to_cpu(self);
                    }
                    self.device = Device::CPU;
                }
            }
            Device::GPU => {
                if self.device == Device::CPU {
                    unsafe {
                        transfer_to_gpu(self);
                    }
                    self.device = Device::GPU;
                }
            }
        }
    }
}

pub fn free_image(image: Image) {
    match image.device {
        Device::CPU => unsafe {
            free_image_cpu(&image as *const Image as *mut Image);
        },
        Device::GPU => unsafe {
            free_image_gpu(&image as *const Image as *mut Image);
        },
    }
}

// Implement Drop trait for Image struct
// impl Drop for Image {
//     fn drop(&mut self) {
//         unsafe {
//             free_image(*self);
//         }
//     }
// }
