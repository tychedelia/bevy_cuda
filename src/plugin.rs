use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::render::RenderApp;

use crate::CudaContext;

#[derive(Default)]
pub struct CudaPlugin {
    pub ordinal: usize,
}

impl Plugin for CudaPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let ordinal = self.ordinal;

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            warn!("CudaPlugin: RenderApp not found");
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();
        let wgpu_device = render_device.wgpu_device();

        match CudaContext::new(wgpu_device, ordinal) {
            Ok(ctx) => {
                info!("CUDA context created on device {}", ordinal);
                render_app.insert_resource(ctx);
            }
            Err(e) => {
                error!("Failed to create CUDA context: {:?}", e);
            }
        }
    }
}
