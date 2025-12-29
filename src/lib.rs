pub mod buffer;
pub mod context;
pub mod error;
pub mod module;
pub mod stream;

mod interop;

#[cfg(feature = "python")]
pub mod python;

pub use buffer::CudaBuffer;
pub use context::CudaContext;
pub use error::{CudaError, Result};
pub use module::{CudaFunction, CudaModule};
pub use stream::CudaStream;

pub use cudarc::driver::{LaunchConfig, PushKernelArg};
pub use cudarc::nvrtc::{compile_ptx, Ptx};

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::render::RenderApp;

#[derive(Default)]
pub struct CudaPlugin {
    pub ordinal: usize,
}

impl CudaPlugin {
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    pub fn cuda_device_count() -> Result<i32> {
        CudaContext::device_count()
    }
}

impl Plugin for CudaPlugin {
    fn build(&self, _app: &mut App) {
        match Self::cuda_device_count() {
            Ok(0) => {
                error!("CudaPlugin: No CUDA-capable devices found");
                return;
            }
            Ok(count) => {
                info!("CudaPlugin: Found {} CUDA device(s)", count);
            }
            Err(e) => {
                error!("CudaPlugin: Failed to query CUDA devices: {}", e);
                return;
            }
        }

        if self.ordinal >= Self::cuda_device_count().unwrap_or(0) as usize {
            error!(
                "CudaPlugin: Device ordinal {} out of range",
                self.ordinal
            );
        }
    }

    fn finish(&self, app: &mut App) {
        let ordinal = self.ordinal;

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            warn!("CudaPlugin: RenderApp not found");
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();
        let wgpu_device = render_device.wgpu_device();

        let is_vulkan = unsafe {
            wgpu_device
                .as_hal::<wgpu_hal::api::Vulkan>()
                .is_some()
        };

        if !is_vulkan {
            error!("CudaPlugin: Vulkan backend required for CUDA interop");
            return;
        }

        match CudaContext::new(wgpu_device, ordinal) {
            Ok(ctx) => {
                let name = ctx.name().unwrap_or_else(|_| "Unknown".to_string());
                let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));
                info!(
                    "CudaPlugin: Context created on {} (sm_{}{})",
                    name, major, minor
                );
                render_app.insert_resource(ctx);
            }
            Err(e) => {
                error!("CudaPlugin: Failed to create context: {:?}", e);
            }
        }
    }
}
