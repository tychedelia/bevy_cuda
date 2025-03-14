use ash::{Device, Instance, khr, vk};
use bevy::utils::default;
use cudarc::driver::{CudaDevice, DevicePtr, DeviceRepr, DeviceSlice, DriverError, result, sys};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::env;
use std::fs::File;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::Range;
use std::os::windows::io::{FromRawHandle, RawHandle};
use std::path::PathBuf;
use std::sync::Arc;

pub fn with_vk_cuda_buffer<F>(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    original_image: vk::Image,
    width: u32,
    height: u32,
    image_format: vk::Format,
    queue_family_index: u32,
    cuda_device_id: i32,
    callback: F,
) -> PyResult<()>
where
    F: FnOnce(Python, Bound<PyAny>) -> PyResult<()>,
{
    let channels = match image_format {
        vk::Format::R8G8B8A8_UNORM => 4,
        vk::Format::R8G8B8_UNORM => 3,
        vk::Format::R8G8B8A8_SRGB => 4,
        _ => 4,
    };

    let buffer_size = (width * height * channels) as u64;

    let mut ext_mem_buffer_info = vk::ExternalMemoryBufferCreateInfoKHR::default();
    ext_mem_buffer_info.s_type = vk::StructureType::EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR;
    ext_mem_buffer_info.handle_types = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;

    let mut buffer_create_info = vk::BufferCreateInfo::default();
    buffer_create_info.s_type = vk::StructureType::BUFFER_CREATE_INFO;
    buffer_create_info.p_next = &ext_mem_buffer_info as *const _ as *const std::ffi::c_void;
    buffer_create_info.size = buffer_size;
    buffer_create_info.usage =
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
    buffer_create_info.sharing_mode = vk::SharingMode::EXCLUSIVE;

    let buffer = unsafe {
        device
            .create_buffer(&buffer_create_info, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create buffer: {:?}", e)))?
    };

    let buffer_mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index =
        find_memory_type_for_external(instance, physical_device, buffer_mem_reqs.memory_type_bits)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to find memory type: {}", e)))?;

    let mut export_mem_info = vk::ExportMemoryAllocateInfo::default();
    export_mem_info.s_type = vk::StructureType::EXPORT_MEMORY_ALLOCATE_INFO;
    export_mem_info.handle_types = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;

    let mut alloc_info = vk::MemoryAllocateInfo::default();
    alloc_info.s_type = vk::StructureType::MEMORY_ALLOCATE_INFO;
    alloc_info.p_next = &export_mem_info as *const _ as *const std::ffi::c_void;
    alloc_info.allocation_size = buffer_mem_reqs.size;
    alloc_info.memory_type_index = memory_type_index;

    let buffer_memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate memory: {:?}", e)))?
    };

    unsafe {
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to bind buffer memory: {:?}", e))
            })?
    };

    let mut command_pool_info = vk::CommandPoolCreateInfo::default();
    command_pool_info.s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = vk::CommandPoolCreateFlags::TRANSIENT;
    command_pool_info.queue_family_index = queue_family_index;

    let command_pool = unsafe {
        device
            .create_command_pool(&command_pool_info, None)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create command pool: {:?}", e))
            })?
    };

    let mut command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default();
    command_buffer_allocate_info.s_type = vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.command_pool = command_pool;
    command_buffer_allocate_info.level = vk::CommandBufferLevel::PRIMARY;
    command_buffer_allocate_info.command_buffer_count = 1;

    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to allocate command buffer: {:?}", e))
            })?
    };
    let command_buffer = command_buffers[0];

    let mut command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
    command_buffer_begin_info.s_type = vk::StructureType::COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.flags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;

    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to begin command buffer: {:?}", e))
            })?
    };

    let mut memory_barrier = vk::MemoryBarrier::default();
    memory_barrier.s_type = vk::StructureType::MEMORY_BARRIER;
    memory_barrier.src_access_mask = vk::AccessFlags::MEMORY_WRITE;
    memory_barrier.dst_access_mask =
        vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[memory_barrier],
            &[],
            &[],
        );
    }

    let mut barrier = vk::ImageMemoryBarrier::default();
    barrier.s_type = vk::StructureType::IMAGE_MEMORY_BARRIER;

    barrier.src_access_mask = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

    barrier.old_layout = vk::ImageLayout::UNDEFINED;
    barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    barrier.src_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    barrier.dst_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    barrier.image = original_image;
    barrier.subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
    barrier.subresource_range.base_mip_level = 0;
    barrier.subresource_range.level_count = 1;
    barrier.subresource_range.base_array_layer = 0;
    barrier.subresource_range.layer_count = 1;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    let mut copy_region = vk::BufferImageCopy::default();
    copy_region.buffer_offset = 0;
    copy_region.buffer_row_length = 0;
    copy_region.buffer_image_height = 0;
    copy_region.image_subresource.aspect_mask = vk::ImageAspectFlags::COLOR;
    copy_region.image_subresource.mip_level = 0;
    copy_region.image_subresource.base_array_layer = 0;
    copy_region.image_subresource.layer_count = 1;
    copy_region.image_offset = vk::Offset3D { x: 0, y: 0, z: 0 };
    copy_region.image_extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };

    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            original_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[copy_region],
        );
    }

    let mut buffer_memory_barrier = vk::BufferMemoryBarrier::default();
    buffer_memory_barrier.s_type = vk::StructureType::BUFFER_MEMORY_BARRIER;
    buffer_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;

    buffer_memory_barrier.dst_access_mask = vk::AccessFlags::MEMORY_READ;
    buffer_memory_barrier.src_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    buffer_memory_barrier.dst_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    buffer_memory_barrier.buffer = buffer;
    buffer_memory_barrier.offset = 0;
    buffer_memory_barrier.size = vk::WHOLE_SIZE;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_memory_barrier],
            &[],
        );
    }

    unsafe {
        device.end_command_buffer(command_buffer).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to end command buffer: {:?}", e))
        })?
    };

    let mut fence_create_info = vk::FenceCreateInfo::default();
    fence_create_info.s_type = vk::StructureType::FENCE_CREATE_INFO;
    fence_create_info.flags = vk::FenceCreateFlags::empty();

    let fence = unsafe {
        device
            .create_fence(&fence_create_info, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create fence: {:?}", e)))?
    };

    let mut submit_info = vk::SubmitInfo::default();
    submit_info.s_type = vk::StructureType::SUBMIT_INFO;
    submit_info.command_buffer_count = 1;
    submit_info.p_command_buffers = &command_buffer;

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    unsafe {
        device
            .queue_submit(queue, &[submit_info], fence)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to submit queue: {:?}", e)))?;

        let result = device.wait_for_fences(&[fence], true, 5_000_000_000);
        if result.is_err() {
            println!("Fence wait timed out!");
        }

        device
            .reset_fences(&[fence])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reset fence: {:?}", e)))?;
        device.destroy_fence(fence, None);
    }

    let ext_mem_win32 = khr::external_memory_win32::Device::new(instance, device);

    let mut handle_info = vk::MemoryGetWin32HandleInfoKHR::default();
    handle_info.s_type = vk::StructureType::MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.memory = buffer_memory;
    handle_info.handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;

    let handle = unsafe {
        ext_mem_win32
            .get_memory_win32_handle(&handle_info)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get Win32 handle: {:?}", e)))?
    };

    let file = unsafe { File::from_raw_handle(handle as RawHandle) };

    let cuda_device = CudaDevice::new(cuda_device_id as usize)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create CUDA device: {}", e)))?;

    let cuda_ext_memory =
        unsafe { cuda_device.import_external_memory_dedicated(file, buffer_size) }.map_err(
            |e| PyRuntimeError::new_err(format!("Failed to import external memory: {}", e)),
        )?;

    let mapped_buffer = cuda_ext_memory
        .map_all()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get mapped buffer: {}", e)))?;

    unsafe {
        device.free_command_buffers(command_pool, &[command_buffer]);
    }

    Python::with_gil(|py| {
        let cuda_array_dict = PyDict::new(py);

        cuda_array_dict.set_item("data", (mapped_buffer.device_ptr as usize, false))?;

        let itemsize = 1;
        let stride_c = itemsize;
        let stride_w = channels as usize * stride_c;
        let stride_h = width as usize * stride_w;

        cuda_array_dict.set_item(
            "shape",
            (height as usize, width as usize, channels as usize),
        )?;

        cuda_array_dict.set_item("strides", (stride_h, stride_w, stride_c))?;

        cuda_array_dict.set_item("typestr", "<u1")?;
        cuda_array_dict.set_item("version", 3)?;

        let locals = PyDict::new(py);
        locals.set_item("__cuda_array_interface__", cuda_array_dict)?;

        let cuda_array_cls = py.eval(
            c"type('CudaArray', (), {'__cuda_array_interface__': __cuda_array_interface__, 'shape': __cuda_array_interface__['shape'], 'device_ptr': __cuda_array_interface__['data'][0]})",
            None,
            Some(&locals),
        )?;

        let cuda_array = cuda_array_cls.call0()?;

        let result = callback(py, cuda_array);

        PyResult::Ok(())
    })?;
    unsafe {
        sys::lib()
            .cuCtxSynchronize()
            .result()
            .expect("Failed to synchronize CUDA context");
    }

    let mut command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default();
    command_buffer_allocate_info.s_type = vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.command_pool = command_pool;
    command_buffer_allocate_info.level = vk::CommandBufferLevel::PRIMARY;
    command_buffer_allocate_info.command_buffer_count = 1;

    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to allocate command buffer: {:?}", e))
            })?
    };
    let command_buffer = command_buffers[0];

    let mut command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
    command_buffer_begin_info.s_type = vk::StructureType::COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.flags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;

    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to begin command buffer: {:?}", e))
            })?
    };

    let mut host_to_device_barrier = vk::BufferMemoryBarrier::default();
    host_to_device_barrier.s_type = vk::StructureType::BUFFER_MEMORY_BARRIER;

    host_to_device_barrier.src_access_mask = vk::AccessFlags::MEMORY_WRITE;

    host_to_device_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
    host_to_device_barrier.src_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    host_to_device_barrier.dst_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    host_to_device_barrier.buffer = buffer;
    host_to_device_barrier.offset = 0;
    host_to_device_barrier.size = vk::WHOLE_SIZE;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[host_to_device_barrier],
            &[],
        );
    }

    let mut barrier = vk::ImageMemoryBarrier::default();
    barrier.s_type = vk::StructureType::IMAGE_MEMORY_BARRIER;
    barrier.src_access_mask = vk::AccessFlags::SHADER_READ;
    barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.src_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    barrier.dst_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    barrier.image = original_image;
    barrier.subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
    barrier.subresource_range.base_mip_level = 0;
    barrier.subresource_range.level_count = 1;
    barrier.subresource_range.base_array_layer = 0;
    barrier.subresource_range.layer_count = 1;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    let copy_region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            original_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
    }

    let mut post_barrier = vk::ImageMemoryBarrier::default();
    post_barrier.s_type = vk::StructureType::IMAGE_MEMORY_BARRIER;
    post_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    post_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
    post_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    post_barrier.new_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    post_barrier.src_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    post_barrier.dst_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
    post_barrier.image = original_image;
    post_barrier.subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
    post_barrier.subresource_range.base_mip_level = 0;
    post_barrier.subresource_range.level_count = 1;
    post_barrier.subresource_range.base_array_layer = 0;
    post_barrier.subresource_range.layer_count = 1;

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[post_barrier],
        );
    }

    unsafe {
        device.end_command_buffer(command_buffer).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to end command buffer: {:?}", e))
        })?
    };

    let mut fence_create_info = vk::FenceCreateInfo::default();
    fence_create_info.s_type = vk::StructureType::FENCE_CREATE_INFO;
    fence_create_info.flags = vk::FenceCreateFlags::empty();

    let fence = unsafe {
        device
            .create_fence(&fence_create_info, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create fence: {:?}", e)))?
    };

    let mut submit_info = vk::SubmitInfo::default();
    submit_info.s_type = vk::StructureType::SUBMIT_INFO;
    submit_info.command_buffer_count = 1;
    submit_info.p_command_buffers = &command_buffer;

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    unsafe {
        device
            .queue_submit(queue, &[submit_info], fence)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to submit queue: {:?}", e)))?;

        let result = device.wait_for_fences(&[fence], true, 5_000_000_000);
        if result.is_err() {
            println!("Fence wait timed out!");
        }
    }

    unsafe {
        device
            .reset_fences(&[fence])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to reset fence: {:?}", e)))?;
        device.destroy_fence(fence, None);

        device.free_command_buffers(command_pool, &[command_buffer]);
        device.destroy_command_pool(command_pool, None);
        device.free_memory(buffer_memory, None);
        device.destroy_buffer(buffer, None);
    }

    Ok(())
}

fn find_memory_type_for_external(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for i in 0..mem_properties.memory_type_count {
        let memory_type = mem_properties.memory_types[i as usize];
        if (type_filter & (1 << i)) != 0
            && (memory_type.property_flags
                & (vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL))
                == (vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return Ok(i);
        }
    }

    for i in 0..mem_properties.memory_type_count {
        let memory_type = mem_properties.memory_types[i as usize];
        if (type_filter & (1 << i)) != 0
            && (memory_type.property_flags
                & (vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE))
                == (vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Ok(i);
        }
    }

    for i in 0..mem_properties.memory_type_count {
        let memory_type = mem_properties.memory_types[i as usize];
        if (type_filter & (1 << i)) != 0
            && (memory_type.property_flags & vk::MemoryPropertyFlags::HOST_VISIBLE)
                == vk::MemoryPropertyFlags::HOST_VISIBLE
        {
            return Ok(i);
        }
    }

    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0 {
            return Ok(i);
        }
    }

    Err("Failed to find suitable memory type for external memory".into())
}

trait CudaDeviceExtensions {
    unsafe fn import_external_memory_dedicated(
        self: &Arc<Self>,
        file: File,
        size: u64,
    ) -> Result<ExternalMemory, DriverError>;
}

impl CudaDeviceExtensions for CudaDevice {
    unsafe fn import_external_memory_dedicated(
        self: &Arc<Self>,
        file: File,
        size: u64,
    ) -> Result<ExternalMemory, DriverError> {
        self.bind_to_thread()?;

        #[cfg(windows)]
        let external_memory = unsafe {
            use std::os::windows::io::AsRawHandle;

            let raw_handle = file.as_raw_handle();

            let mut external_memory = std::mem::MaybeUninit::uninit();
            let handle_description = cudarc::driver::sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
                type_: cudarc::driver::sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
                handle: cudarc::driver::sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
                    win32: cudarc::driver::sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                        handle: raw_handle,
                        name: std::ptr::null(),
                    },
                },
                size,
                flags: 1,
                reserved: [0; 16],
            };

            let res = cudarc::driver::sys::lib()
                .cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description)
                .result()?;

            external_memory.assume_init()
        };

        Ok(ExternalMemory {
            external_memory,
            size,
            device: self.clone(),
            _file: ManuallyDrop::new(file),
        })
    }
}

#[derive(Debug)]
pub struct ExternalMemory {
    external_memory: sys::CUexternalMemory,
    size: u64,
    device: Arc<CudaDevice>,
    _file: ManuallyDrop<File>,
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        self.device.bind_to_thread().unwrap();

        unsafe { result::external_memory::destroy_external_memory(self.external_memory) }.unwrap();

        #[cfg(windows)]
        unsafe {
            ManuallyDrop::<File>::drop(&mut self._file)
        };
    }
}

impl ExternalMemory {
    pub fn map_all(self) -> Result<MappedBuffer, DriverError> {
        let size = self.size as usize;
        self.map_range(0..size)
    }

    pub fn map_range(self, range: Range<usize>) -> Result<MappedBuffer, DriverError> {
        assert!(range.start as u64 <= self.size);
        assert!(range.end as u64 <= self.size);
        let device_ptr = unsafe {
            result::external_memory::get_mapped_buffer(
                self.external_memory,
                range.start as u64,
                range.len() as u64,
            )
        }?;
        Ok(MappedBuffer {
            device_ptr,
            len: range.len(),
            external_memory: self,
        })
    }
}

#[derive(Debug)]
pub struct MappedBuffer {
    device_ptr: sys::CUdeviceptr,
    len: usize,
    external_memory: ExternalMemory,
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        self.external_memory.device.bind_to_thread().unwrap();
        unsafe { result::memory_free(self.device_ptr) }.unwrap()
    }
}

impl DeviceSlice<u8> for MappedBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

impl DevicePtr<u8> for MappedBuffer {
    fn device_ptr(&self) -> &sys::CUdeviceptr {
        &self.device_ptr
    }
}
