use crate::interop::VkCudaBuffer;
use cudarc::driver::sys::CUdeviceptr;

// A buffer that is available from both Cuda and Vulkan contexts
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

    pub(crate) fn as_shared(&self) -> &VkCudaBuffer {
        &self.inner
    }
}

impl bevy::prelude::Resource for CudaBuffer {}
