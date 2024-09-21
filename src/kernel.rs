use std::ffi::c_int;

/// Represents a 2D convolution kernel with its dimensions and data.
pub struct Kernel {
    /// The width of the kernel.
    pub width: c_int,
    /// The height of the kernel.
    pub height: c_int,
    /// The kernel data stored in a flattened 1D vector.
    pub data: Vec<f32>, 
}

impl Kernel {
    /// Creates a new custom kernel with the specified dimensions and data.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the kernel.
    /// * `height` - The height of the kernel.
    /// * `data` - A flattened vector representing the kernel data.
    ///
    /// # Panics
    ///
    /// This function will panic if the length of `data` does not match `width * height`.
    pub fn new(width: c_int, height: c_int, data: Vec<f32>) -> Self {
        if data.len() != (width * height) as usize {
            panic!("Kernel data length does not match specified width and height");
        }
        Kernel {
            width,
            height,
            data,
        }
    }
}

/// Enum that defines the types of kernels available for image processing.
/// 
/// This includes both fixed-size predefined kernels like Gaussian blur and 
/// sharpening, as well as dynamically sized kernels for larger blurring 
/// or sharpening operations.
pub enum KernelType {
    /// A fixed 3x3 Gaussian blur kernel.
    GaussianBlur3x3,
    /// A fixed 3x3 sharpening kernel.
    Sharpen3x3,
    /// A fixed 3x3 Sobel edge detection kernel.
    EdgeDetectionSobel,
    /// A dynamically generated NxN Gaussian blur kernel.
    GaussianBlurNxN(c_int),
    /// A dynamically generated NxN sharpening kernel.
    SharpenNxN(c_int),
}

/// Returns the corresponding kernel for the provided kernel type.
///
/// # Arguments
///
/// * `kernel_type` - The type of kernel to generate. Can be a predefined 3x3 kernel or a custom NxN kernel.
///
/// # Returns
///
/// A `Kernel` struct with the appropriate width, height, and data for the chosen kernel type.
///
/// # Example
///
/// ```rust
/// use img_rcc::kernel::{get_kernel, KernelType};
/// let kernel = get_kernel(KernelType::GaussianBlur3x3);
/// 
/// assert_eq!(kernel.width, 3);
/// assert_eq!(kernel.height, 3);
/// assert_eq!(kernel.data.len(), 9);
/// ```
pub fn get_kernel(kernel_type: KernelType) -> Kernel {
    match kernel_type {
        KernelType::GaussianBlur3x3 => Kernel {
            width: 3,
            height: 3,
            data: vec![
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                4.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
            ],
        },
        KernelType::Sharpen3x3 => Kernel {
            width: 3,
            height: 3,
            data: vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
        },
        KernelType::EdgeDetectionSobel => Kernel {
            width: 3,
            height: 3,
            data: vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
        },
        KernelType::GaussianBlurNxN(size) => {
            let sigma = size as f32 / 6.0; // Standard deviation
            generate_gaussian_kernel(size, sigma)
        }
        KernelType::SharpenNxN(size) => generate_sharpen_kernel(size),
    }
}

/// Generates a Gaussian blur kernel of size NxN with the specified standard deviation (sigma).
///
/// # Arguments
///
/// * `size` - The size of the kernel (must be a positive integer).
/// * `sigma` - The standard deviation for the Gaussian function.
///
/// # Returns
///
/// A `Kernel` struct containing the generated Gaussian blur data.
///
/// The kernel is normalized so that the sum of all elements equals 1.
fn generate_gaussian_kernel(size: c_int, sigma: f32) -> Kernel {
    let mut kernel = vec![0.0; (size * size) as usize];
    let mut sum = 0.0;
    let center = size as f32 / 2.0;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let value = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
            kernel[(y * size + x) as usize] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    for k in &mut kernel {
        *k /= sum;
    }

    Kernel {
        width: size,
        height: size,
        data: kernel,
    }
}

/// Generates a sharpening kernel of size NxN, with the central element increasing
/// in strength based on the kernel size.
///
/// # Arguments
///
/// * `size` - The size of the kernel (must be a positive integer).
///
/// # Returns
///
/// A `Kernel` struct containing the generated sharpening data.
fn generate_sharpen_kernel(size: c_int) -> Kernel {
    let mut kernel = vec![0.0; (size * size) as usize];
    let center = (size * size / 2) as usize;

    for i in 0..kernel.len() {
        if i == center {
            kernel[i] = (size * size) as f32 - 1.0; // Center element
        } else {
            kernel[i] = -1.0 / (size * size) as f32; // Other elements
        }
    }

    Kernel {
        width: size,
        height: size,
        data: kernel,
    }
}
