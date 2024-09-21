pub mod kernel;

use std::ffi::{c_char, c_float, c_int, c_uchar, CString};
use std::ptr;

use kernel::Kernel;

/// Structure representing an image and its metadata.
///
/// # Fields
/// - `image_data`: Holds the actual pixel data and image metadata like width, height, and channels.
/// - `device`: Indicates whether the image is currently stored on the CPU or GPU.
#[cfg(not(any(test, feature = "benchmark")))]
pub struct Image {
    image_data: ImageData,
    device: Device,
}

#[cfg(any(test, feature = "benchmark"))]
pub struct Image {
    pub(crate) image_data: ImageData,
    pub device: Device,
}

/// Struct representing the image data loaded from an image file.
///
/// # Fields
/// - `width`: The width of the image in pixels.
/// - `height`: The height of the image in pixels.
/// - `channels`: The number of color channels (e.g., 3 for RGB, 4 for RGBA).
/// - `data`: A pointer to the image pixel data stored as unsigned characters.
#[repr(C)]
struct ImageData {
    pub width: c_int,
    pub height: c_int,
    pub channels: c_int,
    pub data: *mut c_uchar,
}

/// Enum representing the device where the image is stored, either CPU or GPU.
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

    fn convolve_image_cpu(
        image: *mut ImageData,
        kernel: *const c_float,
        width: c_int,
        height: c_int,
    );
    fn convolve_image_gpu(
        image: *mut ImageData,
        kernel: *const c_float,
        width: c_int,
        height: c_int,
    );

    fn transfer_to_gpu(image: *mut ImageData);
    fn transfer_to_cpu(image: *mut ImageData);
}

// Public API
impl Image {
    /// Load an image from the specified file path into the CPU memory.
    ///
    /// # Parameters
    /// - `file_path`: The file path of the image to load (as a Rust string).
    ///
    /// # Returns
    /// Returns an `Image` object with the image data loaded onto the CPU.
    ///
    /// # Panics
    /// This function will panic if the image cannot be loaded.
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

    /// Load an image from the specified file path into the specified device (CPU or GPU).
    ///
    /// # Parameters
    /// - `file_path`: The file path of the image to load (as a Rust string).
    /// - `device`: The device where the image should be loaded (CPU or GPU).
    ///
    /// # Returns
    /// Returns an `Image` object with the image data loaded onto the specified device.
    ///
    /// # Panics
    /// This function will panic if the image cannot be loaded.
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

    /// Save the image to the specified file path.
    ///
    /// The function will save the image from the device (CPU or GPU) that it currently resides on.
    ///
    /// # Parameters
    /// - `file_path`: The file path where the image should be saved (as a Rust string).
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

    /// Convert the image to grayscale, either on the CPU or GPU, depending on where the image is currently stored.
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

    /// Transfer the image between devices (CPU and GPU).
    ///
    /// # Parameters
    /// - `device`: The target device (CPU or GPU) to which the image should be transferred.
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

    /// Applies the specified convolution kernel to the image, either on the CPU or GPU,
    /// depending on the current device where the image resides.
    ///
    /// This function will choose between CPU or GPU convolution based on the `device` field
    /// of the `Image` struct. If the image is stored in CPU memory, the convolution will
    /// be performed on the CPU. If the image is on the GPU, the convolution will be done
    /// using CUDA on the GPU.
    ///
    /// # Arguments
    ///
    /// * `kernel` - A reference to a `Kernel` struct that defines the kernel data to be applied
    ///              during convolution. The kernel must have valid dimensions and data.
    /// 
    /// This function supports applying both custom and predefined kernels to an image.
    pub fn convolve(&mut self, kernel: &Kernel) {
        match self.device {
            Device::CPU => unsafe {
                convolve_image_cpu(
                    &self.image_data as *const ImageData as *mut ImageData,
                    kernel.data.as_ptr(),
                    kernel.width,
                    kernel.height,
                );
            },
            Device::GPU => unsafe {
                convolve_image_gpu(
                    &self.image_data as *const ImageData as *mut ImageData,
                    kernel.data.as_ptr(),
                    kernel.width,
                    kernel.height,
                );
            },
        }
    }
}

impl Drop for Image {
    /// Custom `Drop` implementation to free image data when the `Image` object goes out of scope.
    /// It ensures that the image data is properly deallocated from either CPU or GPU memory.
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
