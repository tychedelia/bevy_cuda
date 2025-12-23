use crate::interop::VkCudaBuffer;
use cudarc::driver::sys::CUdeviceptr;

pub struct CudaBuffer {
    inner: CudaBufferInner,
}

pub(crate) enum CudaBufferInner {
    Regular { ptr: CUdeviceptr, size: u64 },
    Shared(VkCudaBuffer),
}

impl CudaBuffer {
    pub(crate) fn new(ptr: CUdeviceptr, size: u64) -> Self {
        Self {
            inner: CudaBufferInner::Regular { ptr, size },
        }
    }

    pub(crate) fn from_shared(vk_buffer: VkCudaBuffer) -> Self {
        Self {
            inner: CudaBufferInner::Shared(vk_buffer),
        }
    }

    pub fn ptr(&self) -> CUdeviceptr {
        match &self.inner {
            CudaBufferInner::Regular { ptr, .. } => *ptr,
            CudaBufferInner::Shared(vk_buf) => vk_buf.ptr(),
        }
    }

    pub fn size(&self) -> u64 {
        match &self.inner {
            CudaBufferInner::Regular { size, .. } => *size,
            CudaBufferInner::Shared(vk_buf) => vk_buf.size(),
        }
    }

    pub(crate) fn as_shared(&self) -> Option<&VkCudaBuffer> {
        match &self.inner {
            CudaBufferInner::Shared(vk_buf) => Some(vk_buf),
            CudaBufferInner::Regular { .. } => None,
        }
    }
}

impl bevy::prelude::Resource for CudaBuffer {}
