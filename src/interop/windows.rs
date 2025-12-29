use crate::error::{CudaError, Result};
use crate::interop::external_memory::{ExternalMemory, MappedBuffer};
use ash::{khr, vk, Device, Instance};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::CudaContext;
use std::os::windows::io::RawHandle;
use std::sync::Arc;

pub struct VkCudaBuffer {
    vk_buffer: vk::Buffer,
    vk_memory: vk::DeviceMemory,
    _external_memory: ExternalMemory,
    mapped_buffer: MappedBuffer,
    size: u64,
    device: Device,
}

impl VkCudaBuffer {
    pub fn new(
        instance: &Instance,
        device: Device,
        physical_device: vk::PhysicalDevice,
        cuda_ctx: Arc<CudaContext>,
        size: u64,
    ) -> Result<Self> {
        let mut ext_mem_buffer_info = vk::ExternalMemoryBufferCreateInfoKHR::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);

        let buffer_create_info = vk::BufferCreateInfo::default()
            .push_next(&mut ext_mem_buffer_info)
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vk_buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };

        let mem_reqs = unsafe { device.get_buffer_memory_requirements(vk_buffer) };

        let memory_type_index =
            find_memory_type_for_external(instance, physical_device, mem_reqs.memory_type_bits)?;

        let mut export_mem_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .push_next(&mut export_mem_info)
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index);

        let vk_memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        unsafe { device.bind_buffer_memory(vk_buffer, vk_memory, 0)? };

        let ext_mem_win32 = khr::external_memory_win32::Device::new(instance, &device);

        let handle_info = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(vk_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);

        let handle: RawHandle =
            unsafe { ext_mem_win32.get_memory_win32_handle(&handle_info)? as RawHandle };

        let external_memory =
            unsafe { ExternalMemory::import_win32(cuda_ctx, handle, mem_reqs.size) }
                .map_err(|e| CudaError::Platform(format!("CUDA import failed: {:?}", e)))?;

        let mapped_buffer = external_memory
            .map_all()
            .map_err(|e| CudaError::Platform(format!("CUDA map failed: {:?}", e)))?;

        Ok(Self {
            vk_buffer,
            vk_memory,
            _external_memory: external_memory,
            mapped_buffer,
            size,
            device,
        })
    }

    pub fn device_ptr(&self) -> CUdeviceptr {
        self.mapped_buffer.device_ptr()
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.vk_buffer
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Drop for VkCudaBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.vk_buffer, None);
            self.device.free_memory(self.vk_memory, None);
        }
    }
}

unsafe impl Send for VkCudaBuffer {}
unsafe impl Sync for VkCudaBuffer {}

fn find_memory_type_for_external(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
) -> Result<u32> {
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for i in 0..mem_properties.memory_type_count {
        let memory_type = mem_properties.memory_types[i as usize];
        let required = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        if (type_filter & (1 << i)) != 0 && memory_type.property_flags.contains(required) {
            return Ok(i);
        }
    }

    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0 {
            return Ok(i);
        }
    }

    Err(CudaError::NoSuitableMemoryType)
}
