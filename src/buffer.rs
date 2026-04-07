use crate::interop::VkCudaBuffer;
use cudarc::driver::sys::CUdeviceptr;

// A buffer that is available from both Cuda and Vulkan contexts
#[derive(bevy::prelude::Resource)]
pub struct CudaBuffer {
    inner: VkCudaBuffer,
}

impl CudaBuffer {
    pub(crate) fn from_shared(vk_buffer: VkCudaBuffer) -> Self {
        Self { inner: vk_buffer }
    }

    pub fn device_ptr(&self) -> CUdeviceptr {
        self.inner.device_ptr()
    }

    pub fn size(&self) -> u64 {
        self.inner.size()
    }

    /// Copy data from an arbitrary CUDA device pointer into this buffer.
    pub fn copy_from_device_ptr(&self, src: u64, size: u64) -> crate::error::Result<()> {
        unsafe {
            let result = cudarc::driver::sys::cuMemcpyDtoD_v2(
                self.inner.device_ptr(),
                src,
                size as usize,
            );
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::error::CudaError::Driver(
                    cudarc::driver::DriverError(result),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn as_shared(&self) -> &VkCudaBuffer {
        &self.inner
    }
}

