use crate::buffer::CudaBuffer;
use crate::error::{CudaError, Result};
use crate::interop::VkCudaBuffer;
use crate::module::CudaModule;
use crate::stream::CudaStream;
use ash::{vk, Device, Instance};
use cudarc::driver::CudaContext as CudarcContext;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use wgpu::TextureFormat;

#[derive(bevy::prelude::Resource)]
pub struct CudaContext {
    inner: Arc<CudarcContext>,
    vk_instance: Instance,
    vk_device: Device,
    vk_physical_device: vk::PhysicalDevice,
    vk_queue: vk::Queue,
    vk_queue_family_index: u32,
}

impl CudaContext {
    pub fn new(wgpu_device: &wgpu::Device, ordinal: usize) -> Result<Self> {
        let inner = CudarcContext::new(ordinal)?;

        let (vk_instance, vk_device, vk_physical_device, vk_queue, vk_queue_family_index) =
            unsafe { extract_vulkan_handles(wgpu_device)? };

        Ok(Self {
            inner,
            vk_instance,
            vk_device,
            vk_physical_device,
            vk_queue,
            vk_queue_family_index,
        })
    }

    pub fn device_count() -> Result<i32> {
        Ok(CudarcContext::device_count()?)
    }

    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    pub fn name(&self) -> Result<String> {
        Ok(self.inner.name()?)
    }

    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok(self.inner.compute_capability()?)
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        Arc::new(CudaStream::new(self.inner.default_stream()))
    }

    pub fn new_stream(&self) -> Result<Arc<CudaStream>> {
        let stream = self.inner.new_stream()?;
        Ok(Arc::new(CudaStream::new(stream)))
    }

    pub fn load_module(&self, ptx: Ptx) -> Result<Arc<CudaModule>> {
        let module = self.inner.load_module(ptx)?;
        Ok(Arc::new(CudaModule::new(module)))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.inner.default_stream().synchronize()?;
        Ok(())
    }

    pub fn into_inner(&self) -> &Arc<CudarcContext> {
        &self.inner
    }

    pub fn create_buffer(&self, size: u64) -> Result<CudaBuffer> {
        let vk_buffer = VkCudaBuffer::new(
            &self.vk_instance,
            self.vk_device.clone(),
            self.vk_physical_device,
            self.inner.clone(),
            size,
        )?;
        Ok(CudaBuffer::from_shared(vk_buffer))
    }

    pub fn copy_texture_to_buffer(
        &self,
        texture: &wgpu::Texture,
        buffer: &CudaBuffer,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<()> {
        let vk_buffer = buffer.as_shared();

        let vk_format = wgpu_format_to_vk(format)?;
        let bytes_per_pixel = format_bytes_per_pixel(vk_format);
        let buffer_size = (width * height * bytes_per_pixel) as u64;

        if buffer.size() < buffer_size {
            return Err(CudaError::Platform(format!(
                "Buffer too small: need {} bytes, have {}",
                buffer_size,
                buffer.size()
            )));
        }

        unsafe {
            let vk_image = get_vk_image_from_wgpu(texture)?;

            let command_pool = self.create_command_pool()?;
            let command_buffer = self.allocate_command_buffer(command_pool)?;

            self.begin_command_buffer(command_buffer)?;

            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.vk_device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

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

            self.vk_device.cmd_copy_image_to_buffer(
                command_buffer,
                vk_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk_buffer.vk_buffer(),
                &[copy_region],
            );

            let buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(vk_buffer.vk_buffer())
                .offset(0)
                .size(vk::WHOLE_SIZE);

            self.vk_device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );

            self.vk_device.end_command_buffer(command_buffer)?;

            self.submit_and_wait(command_buffer)?;

            self.vk_device.destroy_command_pool(command_pool, None);
        }

        Ok(())
    }

    pub fn copy_buffer_to_texture(
        &self,
        buffer: &CudaBuffer,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<()> {
        let vk_buffer = buffer.as_shared();

        let vk_format = wgpu_format_to_vk(format)?;
        let bytes_per_pixel = format_bytes_per_pixel(vk_format);
        let buffer_size = (width * height * bytes_per_pixel) as u64;

        if buffer.size() < buffer_size {
            return Err(CudaError::Platform(format!(
                "Buffer too small: need {} bytes, have {}",
                buffer_size,
                buffer.size()
            )));
        }

        self.inner
            .default_stream()
            .synchronize()
            .map_err(|e| CudaError::Platform(format!("CUDA sync failed: {:?}", e)))?;

        unsafe {
            let vk_image = get_vk_image_from_wgpu(texture)?;

            let command_pool = self.create_command_pool()?;
            let command_buffer = self.allocate_command_buffer(command_pool)?;

            self.begin_command_buffer(command_buffer)?;

            let buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(vk_buffer.vk_buffer())
                .offset(0)
                .size(vk::WHOLE_SIZE);

            self.vk_device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );

            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.vk_device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

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

            self.vk_device.cmd_copy_buffer_to_image(
                command_buffer,
                vk_buffer.vk_buffer(),
                vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );

            self.vk_device.end_command_buffer(command_buffer)?;

            self.submit_and_wait(command_buffer)?;

            self.vk_device.destroy_command_pool(command_pool, None);
        }

        Ok(())
    }

    unsafe fn create_command_pool(&self) -> Result<vk::CommandPool> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.vk_queue_family_index);

        self.vk_device
            .create_command_pool(&pool_info, None)
            .map_err(CudaError::Vulkan)
    }

    unsafe fn allocate_command_buffer(&self, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = self
            .vk_device
            .allocate_command_buffers(&alloc_info)
            .map_err(CudaError::Vulkan)?;

        Ok(buffers[0])
    }

    unsafe fn begin_command_buffer(&self, cmd: vk::CommandBuffer) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.vk_device
            .begin_command_buffer(cmd, &begin_info)
            .map_err(CudaError::Vulkan)
    }

    unsafe fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<()> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = self
            .vk_device
            .create_fence(&fence_info, None)
            .map_err(CudaError::Vulkan)?;

        let cmd_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

        self.vk_device
            .queue_submit(self.vk_queue, &[submit_info], fence)
            .map_err(CudaError::Vulkan)?;

        self.vk_device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(CudaError::Vulkan)?;

        self.vk_device.destroy_fence(fence, None);

        Ok(())
    }
}

unsafe fn extract_vulkan_handles(
    wgpu_device: &wgpu::Device,
) -> Result<(Instance, Device, vk::PhysicalDevice, vk::Queue, u32)> {
    let hal_device = wgpu_device
        .as_hal::<wgpu_hal::api::Vulkan>()
        .ok_or(CudaError::NoInterop)?;

    let instance = hal_device.shared_instance().raw_instance().clone();
    let vk_device = hal_device.raw_device().clone();
    let physical_device = hal_device.raw_physical_device();

    let queue_family_index = 0u32; //
    let queue = vk_device.get_device_queue(queue_family_index, 0);

    Ok((
        instance,
        vk_device,
        physical_device,
        queue,
        queue_family_index,
    ))
}

unsafe fn get_vk_image_from_wgpu(texture: &wgpu::Texture) -> Result<vk::Image> {
    let hal_texture = texture
        .as_hal::<wgpu_hal::api::Vulkan>()
        .ok_or(CudaError::NoInterop)?;

    Ok(hal_texture.raw_handle())
}

fn wgpu_format_to_vk(format: TextureFormat) -> Result<vk::Format> {
    let vk_format = match format {
        TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        TextureFormat::R8Snorm => vk::Format::R8_SNORM,
        TextureFormat::R8Uint => vk::Format::R8_UINT,
        TextureFormat::R8Sint => vk::Format::R8_SINT,
        TextureFormat::R16Uint => vk::Format::R16_UINT,
        TextureFormat::R16Sint => vk::Format::R16_SINT,
        TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        TextureFormat::Rg8Snorm => vk::Format::R8G8_SNORM,
        TextureFormat::Rg8Uint => vk::Format::R8G8_UINT,
        TextureFormat::Rg8Sint => vk::Format::R8G8_SINT,
        TextureFormat::R32Uint => vk::Format::R32_UINT,
        TextureFormat::R32Sint => vk::Format::R32_SINT,
        TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        TextureFormat::Rg16Uint => vk::Format::R16G16_UINT,
        TextureFormat::Rg16Sint => vk::Format::R16G16_SINT,
        TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Rgba8Snorm => vk::Format::R8G8B8A8_SNORM,
        TextureFormat::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
        TextureFormat::Rgba8Sint => vk::Format::R8G8B8A8_SINT,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::Rg32Uint => vk::Format::R32G32_UINT,
        TextureFormat::Rg32Sint => vk::Format::R32G32_SINT,
        TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        TextureFormat::Rgba16Uint => vk::Format::R16G16B16A16_UINT,
        TextureFormat::Rgba16Sint => vk::Format::R16G16B16A16_SINT,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        TextureFormat::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        _ => {
            return Err(CudaError::Platform(format!(
                "Unsupported texture format: {:?}",
                format
            )))
        }
    };
    Ok(vk_format)
}

fn format_bytes_per_pixel(format: vk::Format) -> u32 {
    match format {
        vk::Format::R8_UNORM | vk::Format::R8_SNORM | vk::Format::R8_UINT | vk::Format::R8_SINT => {
            1
        }
        vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT => 2,
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R32_SFLOAT => 4,
        vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R32G32_UINT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32_SFLOAT => 8,
        vk::Format::R32G32B32A32_UINT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R32G32B32A32_SFLOAT => 16,
        _ => 4,
    }
}
