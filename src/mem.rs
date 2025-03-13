use std::env;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::exceptions::PyRuntimeError;
use std::fs::File;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::Range;
use std::os::windows::io::{FromRawHandle, RawHandle};
use std::path::PathBuf;
use std::sync::Arc;
use ash::{khr, vk, Device, Instance};
use cudarc::driver::{result, sys, CudaDevice, DevicePtr, DeviceRepr, DeviceSlice, DriverError};
use cudarc::driver::sys::{CUmemAllocationHandleType, CUmemAllocationType, CUmemAllocationType_enum, CUmemLocationType, CUmemLocationType_enum};
use cudarc::driver::sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
use cudarc::driver::sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM;

struct VkCudaBuffer<'a> {
    vk_device: &'a Device,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    command_pool: vk::CommandPool,
    mapped_buffer: MappedBuffer,
}

impl<'a> Drop for VkCudaBuffer<'a> {
    fn drop(&mut self) {
        // Clean up in reverse order of creation
        unsafe {
            self.vk_device.destroy_command_pool(self.command_pool, None);
            self.vk_device.destroy_buffer(self.buffer, None);
            self.vk_device.free_memory(self.buffer_memory, None);
        }
    }
}

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
    // Calculate channels based on format
    let channels = match image_format {
        vk::Format::R8G8B8A8_UNORM => 4,
        vk::Format::R8G8B8_UNORM => 3,
        _ => 4, // Default to 4 channels
    };

    // Calculate buffer size
    let buffer_size = (width * height * channels) as usize;

    // STEP 1: Allocate CUDA memory first
    // This is similar to the allocate_shared_cuda_memory function in the other library

    // Create CUDA device
    let cuda_device = CudaDevice::new(cuda_device_id as usize)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create CUDA device: {}", e)))?;

    let cuda_device_arc = Arc::new(cuda_device);

    // Get the handle type for the platform
    let share_type = if cfg!(target_os = "windows") {
        CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32 // CU_MEM_HANDLE_TYPE_WIN32
    } else {
        CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR // CU_MEM_HANDLE_TYPE_FD
    };

    // Create properties for the memory allocation
    let location = cudarc::driver::sys::CUmemLocation {
        type_: CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE, // CU_MEM_LOCATION_TYPE_DEVICE
        id: cuda_device_id,
    };

    let mut prop = cudarc::driver::sys::CUmemAllocationProp {
        type_: CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED, // CU_MEM_ALLOCATION_TYPE_PINNED
        requestedHandleTypes: share_type,
        location,
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: cudarc::driver::sys::CUmemAllocationProp_st__bindgen_ty_1::default(),
    };

    #[cfg(target_os = "windows")]
    {
        use windows::Win32::Security::{ *, Authorization::* };
        use windows::Win32::Foundation::*;
        use windows::Wdk::Foundation::*;
        use windows::core::*;

        // Set up security attributes for Windows
        let sddl = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)\0";
        let mut sec_desc = PSECURITY_DESCRIPTOR::default();
        let result = unsafe { ConvertStringSecurityDescriptorToSecurityDescriptorA(
            PCSTR::from_raw(sddl.as_ptr()),
            SDDL_REVISION_1,
            &mut sec_desc,
            None
        )};

        if result.is_ok() {
            let obj_attributes = OBJECT_ATTRIBUTES {
                Length: std::mem::size_of::<OBJECT_ATTRIBUTES>() as u32,
                RootDirectory: HANDLE::default(),
                ObjectName: std::ptr::null_mut(),
                Attributes: 0,
                SecurityDescriptor: sec_desc.0,
                SecurityQualityOfService: std::ptr::null_mut(),
            };

            prop.win32HandleMetaData = &obj_attributes as *const _ as *mut _;
        } else {
            return Err(PyRuntimeError::new_err("Failed to set up security descriptor for CUDA memory allocation"));
        }
    }

    // Get allocation granularity
    let mut granularity = 0usize;
    unsafe {
        cudarc::driver::sys::lib().cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        ).result()
    }.map_err(|e| PyRuntimeError::new_err(format!("Failed to get allocation granularity: {:?}", e)))?;

    // Align size to granularity
    let aligned_size = ((buffer_size + granularity - 1) / granularity) * granularity;

    // Reserve address space
    let mut device_ptr = 0u64;
    unsafe {
        cudarc::driver::sys::lib().cuMemAddressReserve(
            &mut device_ptr,
            aligned_size,
            granularity,
            0,
            0
        ).result()
    }.map_err(|e| PyRuntimeError::new_err(format!("Failed to reserve CUDA address space: {:?}", e)))?;

    // Create memory
    let mut mem_handle = 0u64;
    unsafe {
        cudarc::driver::sys::lib().cuMemCreate(
            &mut mem_handle,
            aligned_size,
            &prop,
            0
        ).result()
    }.map_err(|e| PyRuntimeError::new_err(format!("Failed to create CUDA memory: {:?}", e)))?;

    // Export handle
    let mut shared_handle = 0isize;
    unsafe {
        cudarc::driver::sys::lib().cuMemExportToShareableHandle(
            &mut shared_handle as *mut isize as *mut _,
            mem_handle,
            share_type,
            0
        ).result()
    }.map_err(|e| PyRuntimeError::new_err(format!("Failed to export CUDA memory handle: {:?}", e)))?;

    println!("Created CUDA memory at: 0x{:x}, size: {}, aligned size: {}", device_ptr, buffer_size, aligned_size);

    // Map the memory
    unsafe {
        cudarc::driver::sys::lib().cuMemMap(
            device_ptr,
            aligned_size,
            0,
            mem_handle,
            0
        ).result()
    }.map_err(|e| PyRuntimeError::new_err(format!("Failed to map CUDA memory: {:?}", e)))?;

    // Release handle (memory will persist until unmapped)
    unsafe {
        cudarc::driver::sys::lib().cuMemRelease(mem_handle).result()
    }.map_err(|e| PyRuntimeError::new_err(format!("Failed to release CUDA memory handle: {:?}", e)))?;

    // Set access permissions
    let access_desc = cudarc::driver::sys::CUmemAccessDesc_st {
        location,
        flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE, // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    };

    unsafe {
        cudarc::driver::sys::lib().cuMemSetAccess(
            device_ptr,
            aligned_size,
            &access_desc,
            1
        ).result()
    }.map_err(|e| PyRuntimeError::new_err(format!("Failed to set CUDA memory access: {:?}", e)))?;

    // STEP 2: Create Vulkan buffer using the exported handle

    // Get the handle type for Vulkan
    let handle_type = if cfg!(target_os = "windows") {
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32
    } else {
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
    };

    // Create buffer with external memory info
    let mut ext_buffer_info = vk::ExternalMemoryBufferCreateInfoKHR::default();
    ext_buffer_info.s_type = vk::StructureType::EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR;
    ext_buffer_info.handle_types = handle_type;

    let mut buffer_create_info = vk::BufferCreateInfo::default();
    buffer_create_info.s_type = vk::StructureType::BUFFER_CREATE_INFO;
    buffer_create_info.p_next = &ext_buffer_info as *const _ as *const std::ffi::c_void;
    buffer_create_info.size = aligned_size as u64;
    buffer_create_info.usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
    buffer_create_info.sharing_mode = vk::SharingMode::EXCLUSIVE;

    let buffer = unsafe {
        device.create_buffer(&buffer_create_info, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Vulkan buffer: {:?}", e)))?
    };

    // Get memory requirements
    let buffer_mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    // Find suitable memory type
    let memory_type_index = find_memory_type_for_external(
        instance,
        physical_device,
        buffer_mem_reqs.memory_type_bits,
    ).map_err(|e| PyRuntimeError::new_err(format!("Failed to find memory type: {}", e)))?;

    // Prepare to import the handle
    #[cfg(target_os = "windows")]
    let mut import_memory_info = vk::ImportMemoryWin32HandleInfoKHR::default()
        .handle_type(handle_type)
        .handle(shared_handle);

    #[cfg(target_os = "linux")]
    let mut import_memory_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(handle_type)
        .fd(shared_handle as std::ffi::c_int);

    // Allocate memory with import info
    let mut alloc_info = vk::MemoryAllocateInfo::default();
    alloc_info.s_type = vk::StructureType::MEMORY_ALLOCATE_INFO;
    alloc_info.p_next = unsafe { &import_memory_info as *const _ as *const std::ffi::c_void };
    alloc_info.allocation_size = buffer_mem_reqs.size;
    alloc_info.memory_type_index = memory_type_index;

    let buffer_memory = unsafe {
        device.allocate_memory(&alloc_info, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate Vulkan memory: {:?}", e)))?
    };

    // Bind memory to buffer
    unsafe {
        device.bind_buffer_memory(buffer, buffer_memory, 0)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to bind buffer memory: {:?}", e)))?
    };

    // STEP 3: Use command buffer to copy image data to the shared buffer

    // Create command pool
    let mut command_pool_info = vk::CommandPoolCreateInfo::default();
    command_pool_info.s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = vk::CommandPoolCreateFlags::TRANSIENT;
    command_pool_info.queue_family_index = queue_family_index;

    let command_pool = unsafe {
        device.create_command_pool(&command_pool_info, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create command pool: {:?}", e)))?
    };

    // Allocate command buffer
    let mut command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default();
    command_buffer_allocate_info.s_type = vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.command_pool = command_pool;
    command_buffer_allocate_info.level = vk::CommandBufferLevel::PRIMARY;
    command_buffer_allocate_info.command_buffer_count = 1;

    let command_buffers = unsafe {
        device.allocate_command_buffers(&command_buffer_allocate_info)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate command buffer: {:?}", e)))?
    };
    let command_buffer = command_buffers[0];

    // Begin command buffer
    let mut command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
    command_buffer_begin_info.s_type = vk::StructureType::COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.flags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;

    unsafe {
        device.begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to begin command buffer: {:?}", e)))?
    };

    // Transition image layout for transfer
    let current_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

    let mut barrier = vk::ImageMemoryBarrier::default();
    barrier.s_type = vk::StructureType::IMAGE_MEMORY_BARRIER;
    barrier.src_access_mask = vk::AccessFlags::SHADER_READ;
    barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
    barrier.old_layout = current_layout;
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
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    // Copy image to buffer
    let mut copy_region = vk::BufferImageCopy::default();
    copy_region.buffer_offset = 0;
    copy_region.buffer_row_length = 0; // Tightly packed
    copy_region.buffer_image_height = 0; // Tightly packed
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

    // Transition image back to original layout
    let mut post_barrier = vk::ImageMemoryBarrier::default();
    post_barrier.s_type = vk::StructureType::IMAGE_MEMORY_BARRIER;
    post_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
    post_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
    post_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    post_barrier.new_layout = current_layout;
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
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[post_barrier],
        );
    }

    // End command buffer recording
    unsafe {
        device.end_command_buffer(command_buffer)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to end command buffer: {:?}", e)))?
    };

    // Submit command buffer
    let mut submit_info = vk::SubmitInfo::default();
    submit_info.s_type = vk::StructureType::SUBMIT_INFO;
    submit_info.command_buffer_count = 1;
    submit_info.p_command_buffers = &command_buffer;

    let queue = unsafe {
        device.get_device_queue(queue_family_index, 0)
    };

    unsafe {
        device.queue_submit(queue, &[submit_info], vk::Fence::null())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to submit queue: {:?}", e)))?;
        device.queue_wait_idle(queue)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to wait for queue idle: {:?}", e)))?;
    }

    // STEP 4: Create a Python object with the CUDA array interface

    // Initialize Python
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        // Setup Python path for importing torch if needed
        let venv_path = match env::var("VIRTUAL_ENV") {
            Ok(path) => path,
            Err(_) => {
                eprintln!("VIRTUAL_ENV not set. Are you running from an activated virtual environment?");
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "VIRTUAL_ENV environment variable not found",
                ));
            }
        };

        println!("Using virtual environment: {}", venv_path);

        let site_packages = if cfg!(windows) {
            format!("{}\\Lib\\site-packages", venv_path)
        } else {
            format!("{}/lib/python3.11/site-packages", venv_path)
        };

        // Add site-packages to Python path
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (site_packages,))?;

        let cuda_array_dict = PyDict::new(py);

        // Set required fields
        cuda_array_dict.set_item("data", (device_ptr as usize, false))?;
        cuda_array_dict.set_item("shape", (height as usize, width as usize, channels as usize))?;

        // Calculate strides for RGBA8_UNORM (uint8 bytes per channel)
        let itemsize = 1; // 1 byte per channel for uint8
        let stride_c = itemsize;
        let stride_w = stride_c * channels as usize;
        let stride_h = stride_w * width as usize;

        cuda_array_dict.set_item("strides", (stride_h, stride_w, stride_c))?;

        // Set correct data type for RGBA8_UNORM
        cuda_array_dict.set_item("typestr", "<u1")?; // Little-endian uint8
        cuda_array_dict.set_item("version", 3)?; // Version 3 of the CUDA array interface

        // Create a simple Python class with __cuda_array_interface__ property
        let locals = PyDict::new(py);
        locals.set_item("__cuda_array_interface__", cuda_array_dict)?;

        let cuda_array_cls = py.eval(
            c"type('CudaArray', (), {'__cuda_array_interface__': __cuda_array_interface__, 'shape': __cuda_array_interface__['shape'], 'device_ptr': __cuda_array_interface__['data'][0]})",
            None,
            Some(&locals),
        )?;

        let cuda_array = cuda_array_cls.call0()?;

        // Call the user's callback with our CUDA array
        let result = callback(py, cuda_array);

        Python::with_gil(|py| {
            // After callback completes
            py.run(c"import gc; gc.collect()", None, None)?; // Force Python garbage collection

            // Force CUDA synchronization through Python
            py.run(c"import torch; torch.cuda.synchronize()", None, None)?;

            PyResult::Ok(())
        })?;

        // Make sure we finish all CUDA operations before cleaning up
        unsafe {
            cudarc::driver::sys::lib().cuCtxSynchronize().result()
        }.map_err(|e| PyRuntimeError::new_err(format!("Failed to synchronize CUDA device: {:?}", e)))?;

        result
    })?;

    unsafe {
        // Create a command buffer for the copy operation
        let mut command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default();
        command_buffer_allocate_info.s_type = vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_allocate_info.command_pool = command_pool;
        command_buffer_allocate_info.level = vk::CommandBufferLevel::PRIMARY;
        command_buffer_allocate_info.command_buffer_count = 1;

        let command_buffers = device.allocate_command_buffers(&command_buffer_allocate_info)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate command buffer: {:?}", e)))?;
        let command_buffer = command_buffers[0];

        // Begin command buffer
        let mut command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
        command_buffer_begin_info.s_type = vk::StructureType::COMMAND_BUFFER_BEGIN_INFO;
        command_buffer_begin_info.flags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;

        device.begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to begin command buffer: {:?}", e)))?;

        // Transition image layout for writing
        let mut barrier = vk::ImageMemoryBarrier::default();
        barrier.s_type = vk::StructureType::IMAGE_MEMORY_BARRIER;
        barrier.src_access_mask = vk::AccessFlags::SHADER_READ;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.old_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL; // Assuming this is the current layout
        barrier.new_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.src_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
        barrier.dst_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
        barrier.image = original_image;
        barrier.subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
        barrier.subresource_range.base_mip_level = 0;
        barrier.subresource_range.level_count = 1;
        barrier.subresource_range.base_array_layer = 0;
        barrier.subresource_range.layer_count = 1;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );

        // Copy buffer to image
        let mut copy_region = vk::BufferImageCopy::default();
        copy_region.buffer_offset = 0;
        copy_region.buffer_row_length = 0; // Tightly packed
        copy_region.buffer_image_height = 0; // Tightly packed
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

        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            original_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );

        // Transition image back to original layout
        let mut post_barrier = vk::ImageMemoryBarrier::default();
        post_barrier.s_type = vk::StructureType::IMAGE_MEMORY_BARRIER;
        post_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        post_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
        post_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        post_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        post_barrier.src_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
        post_barrier.dst_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
        post_barrier.image = original_image;
        post_barrier.subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
        post_barrier.subresource_range.base_mip_level = 0;
        post_barrier.subresource_range.level_count = 1;
        post_barrier.subresource_range.base_array_layer = 0;
        post_barrier.subresource_range.layer_count = 1;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[post_barrier],
        );

        // End command buffer recording
        device.end_command_buffer(command_buffer)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to end command buffer: {:?}", e)))?;

        // Submit command buffer
        let mut submit_info = vk::SubmitInfo::default();
        submit_info.s_type = vk::StructureType::SUBMIT_INFO;
        submit_info.command_buffer_count = 1;
        submit_info.p_command_buffers = &command_buffer;

        let queue = device.get_device_queue(queue_family_index, 0);

        device.queue_submit(queue, &[submit_info], vk::Fence::null())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to submit queue: {:?}", e)))?;
        device.queue_wait_idle(queue)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to wait for queue idle: {:?}", e)))?;

        // Free the command buffer
        device.free_command_buffers(command_pool, &[command_buffer]);
    }

    unsafe {
        // Make sure all CUDA operations are complete
        println!("Synchronizing CUDA device...");
        cudarc::driver::sys::lib().cuCtxSynchronize().result()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to synchronize CUDA: {:?}", e)))?;

        // Ensure we've cleaned up all resources
        println!("Cleaning up resources...");

        // First clean up CUDA resources
        cudarc::driver::sys::lib().cuMemUnmap(device_ptr, aligned_size).result()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to unmap CUDA memory: {:?}", e)))?;

        cudarc::driver::sys::lib().cuMemAddressFree(device_ptr, aligned_size).result()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to free CUDA address: {:?}", e)))?;

        // Then clean up Vulkan resources
        device.destroy_command_pool(command_pool, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(buffer_memory, None);

        println!("Cleanup complete");
    }
    Ok(())
}

fn find_memory_type_for_external(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
) -> Result<u32, Box<dyn std::error::Error>> {
    let mem_properties = unsafe {
        instance.get_physical_device_memory_properties(physical_device)
    };

    // First try to find a memory type that is both DEVICE_LOCAL and HOST_VISIBLE
    for i in 0..mem_properties.memory_type_count {
        let memory_type = mem_properties.memory_types[i as usize];
        if (type_filter & (1 << i)) != 0 &&
            (memory_type.property_flags & (vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE))
                == (vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE) {
            return Ok(i);
        }
    }

    // Fall back to just DEVICE_LOCAL if the above isn't available
    for i in 0..mem_properties.memory_type_count {
        let memory_type = mem_properties.memory_types[i as usize];
        if (type_filter & (1 << i)) != 0 &&
            (memory_type.property_flags & vk::MemoryPropertyFlags::DEVICE_LOCAL)
                == vk::MemoryPropertyFlags::DEVICE_LOCAL {
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

            // Get the raw handle from the file
            let raw_handle = file.as_raw_handle();

            // Create our own version of import_external_memory_opaque_win32 with the dedicated flag
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
                flags: 1, // CUDA_EXTERNAL_MEMORY_DEDICATED = 1
                reserved: [0; 16],
            };

            // Call the CUDA driver function directly
            let res = cudarc::driver::sys::lib()
                .cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description)
                .result()?;

            external_memory.assume_init()
        };

        // Return an ExternalMemory struct just like the original function
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

        // From [CUDA docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735),
        // when successfully importing UNIX file descriptor:
        //
        // > Ownership of the file descriptor is transferred to the CUDA driver when the handle is imported successfully.
        // > Performing any operations on the file descriptor after it is imported results in undefined behavior.
        //
        // On the other hand, on Windows:
        //
        // > Ownership of this handle is not transferred to CUDA after the import operation,
        // > so the application must release the handle using the appropriate system call.
        //
        // Therefore, we manually drop the file when we are on Windows.
        #[cfg(windows)]
        unsafe {
            ManuallyDrop::<File>::drop(&mut self._file)
        };
    }
}


impl ExternalMemory {
    /// Map the whole external memory to get mapped buffer.
    pub fn map_all(self) -> Result<MappedBuffer, DriverError> {
        let size = self.size as usize;
        self.map_range(0..size)
    }

    /// Map a range of the external memory to a mapped buffer.
    ///
    /// Only one mapped buffer is allowed at a time.
    /// This is more restrictive than it necessarily needs to be,
    /// but it makes enforcing safety easier.
    ///
    /// # Panics
    /// This function will panic if the range is invalid,
    /// such as when the start or end is larger than the size.
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

/// An abstraction for a mapped buffer for some external memory.
///
/// This struct can be created via [`cudarc::driver::ExternalMemory::map_range`] or [`cudarc::driver::ExternalMemory::map_all`].
/// The underlying mapped buffer will be freed when this struct is dropped.
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