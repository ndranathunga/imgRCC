// #[cfg(feature = "benchmark")] // currently removed because no need for specific benchmarking codes
// pub mod benchmark;

use std::ffi::{c_char, CString};
use std::os::raw::{c_int, c_uchar};
use std::ptr;

// Define Image struct in Rust matching C++ Image structure
#[cfg(test)]
pub struct Image {
    pub image_data: ImageData,
    pub device: Device,
}

#[cfg(not(test))]
pub struct Image {
    image_data: ImageData,
    device: Device,
}

#[repr(C)]
pub struct ImageData {
    pub width: c_int,
    pub height: c_int,
    pub channels: c_int,
    pub data: *mut c_uchar,
}

#[derive(PartialEq, Debug)]
#[repr(C)]
pub enum Device {
    CPU,
    GPU,
}

extern "C" {
    fn load_image_cpu(file_path: *const c_char) -> *mut ImageData;
    fn save_image_cpu(file_path: *const c_char, image: *mut ImageData);
    fn free_image_cpu(image: *mut ImageData);

    fn load_image_gpu(file_path: *const c_char) -> *mut ImageData;
    fn save_image_gpu(file_path: *const c_char, image: *mut ImageData);
    fn free_image_gpu(image: *mut ImageData);

    fn convert_to_grayscale_cpu(image: *mut ImageData);
    fn convert_to_grayscale_gpu(image: *mut ImageData);

    fn transfer_to_gpu(image: *mut ImageData);
    fn transfer_to_cpu(image: *mut ImageData);
}

// Public API

impl Image {
    pub fn load(file_path: &str) -> Image {
        let c_file_path = CString::new(file_path).expect("CString::new failed");
        let image_ptr = unsafe { load_image_cpu(c_file_path.as_ptr()) };
        if image_ptr.is_null() {
            panic!("Failed to load image.");
        }
        let image_data = unsafe { ptr::read(image_ptr) };

        Image {
            image_data,
            device: Device::CPU,
        }
    }

    pub fn load_to_device(file_path: &str, device: Device) -> Image {
        let c_file_path = CString::new(file_path).expect("CString::new failed");

        let image_data = match device {
            Device::CPU => {
                let image_ptr = unsafe { load_image_cpu(c_file_path.as_ptr()) };
                if image_ptr.is_null() {
                    panic!("Failed to load image on CPU.");
                }
                unsafe { ptr::read(image_ptr) }
            }
            Device::GPU => {
                let image_ptr = unsafe { load_image_gpu(c_file_path.as_ptr()) };
                if image_ptr.is_null() {
                    panic!("Failed to load image on CPU.");
                }
                unsafe { ptr::read(image_ptr) }
            }
        };
        Image { image_data, device }
    }

    pub fn save(&self, file_path: &str) {
        let c_file_path = CString::new(file_path).expect("CString::new failed");

        match self.device {
            Device::CPU => unsafe {
                save_image_cpu(
                    c_file_path.as_ptr(),
                    &self.image_data as *const ImageData as *mut ImageData,
                );
            },
            Device::GPU => unsafe {
                save_image_gpu(
                    c_file_path.as_ptr(),
                    &self.image_data as *const ImageData as *mut ImageData,
                );
            },
        }
    }

    pub fn grayscale(&mut self) {
        match self.device {
            Device::CPU => unsafe {
                convert_to_grayscale_cpu(&self.image_data as *const ImageData as *mut ImageData);
            },
            Device::GPU => unsafe {
                convert_to_grayscale_gpu(&self.image_data as *const ImageData as *mut ImageData);
            },
        }
    }

    pub fn to(&mut self, device: Device) {
        match device {
            Device::CPU => {
                if self.device == Device::GPU {
                    unsafe {
                        transfer_to_cpu(&self.image_data as *const ImageData as *mut ImageData);
                    }
                    self.device = Device::CPU;
                }
            }
            Device::GPU => {
                if self.device == Device::CPU {
                    unsafe {
                        transfer_to_gpu(&self.image_data as *const ImageData as *mut ImageData);
                    }
                    self.device = Device::GPU;
                }
            }
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        match self.device {
            Device::CPU => unsafe {
                free_image_cpu(&self.image_data as *const ImageData as *mut ImageData);
            },
            Device::GPU => unsafe {
                free_image_gpu(&self.image_data as *const ImageData as *mut ImageData);
            },
        }
    }
}
