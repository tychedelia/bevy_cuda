use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),

    #[error("Platform error: {0}")]
    Platform(String),

    #[error("wgpu interop not available")]
    NoInterop,

    #[error("No suitable memory type found for external memory")]
    NoSuitableMemoryType,
}

pub type Result<T> = std::result::Result<T, CudaError>;
