use cudarc::driver::result::external_memory as cuda_ext;
use cudarc::driver::sys::{CUdeviceptr, CUexternalMemory};
use cudarc::driver::{DriverError, CudaContext};
use std::ops::Range;
use std::sync::Arc;

#[cfg(windows)]
use std::os::windows::io::RawHandle;

pub struct ExternalMemory {
    handle: CUexternalMemory,
    size: u64,
    ctx: Arc<CudaContext>,
}

impl ExternalMemory {
    #[cfg(windows)]
    pub unsafe fn import_win32(
        ctx: Arc<CudaContext>,
        handle: RawHandle,
        size: u64,
    ) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let external_memory = cuda_ext::import_external_memory_opaque_win32(handle, size)?;
        Ok(Self {
            handle: external_memory,
            size,
            ctx,
        })
    }

    #[cfg(unix)]
    pub unsafe fn import_fd(
        ctx: Arc<CudaContext>,
        fd: std::os::unix::io::RawFd,
        size: u64,
    ) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let external_memory = cuda_ext::import_external_memory_opaque_fd(fd, size)?;
        Ok(Self {
            handle: external_memory,
            size,
            ctx,
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn map_all(&self) -> Result<MappedBuffer, DriverError> {
        self.map_range(0..self.size as usize)
    }

    pub fn map_range(&self, range: Range<usize>) -> Result<MappedBuffer, DriverError> {
        assert!(range.start as u64 <= self.size);
        assert!(range.end as u64 <= self.size);

        self.ctx.bind_to_thread()?;

        let device_ptr = unsafe {
            cuda_ext::get_mapped_buffer(self.handle, range.start as u64, range.len() as u64)?
        };

        Ok(MappedBuffer {
            device_ptr,
            len: range.len(),
            ctx: self.ctx.clone(),
        })
    }
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = cuda_ext::destroy_external_memory(self.handle);
        }
    }
}

pub struct MappedBuffer {
    device_ptr: CUdeviceptr,
    len: usize,
    ctx: Arc<CudaContext>,
}

impl MappedBuffer {
    pub fn ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = cudarc::driver::sys::cuMemFree_v2(self.device_ptr).result();
        }
    }
}

unsafe impl Send for ExternalMemory {}
unsafe impl Sync for ExternalMemory {}
unsafe impl Send for MappedBuffer {}
unsafe impl Sync for MappedBuffer {}
