mod mem;

use ash::vk;
use bevy::asset::{AssetLoader, RenderAssetUsages};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::renderer::{RenderAdapter, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    ecs::query::QueryItem,
    prelude::*,
    render::{
        RenderApp,
        extract_component::{
            DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::*,
        renderer::RenderContext,
        view::ViewTarget,
    },
};
use lazy_static::lazy_static;
use pyo3::prelude::{PyAnyMethods, PyDictMethods, PyModule};
use pyo3::{IntoPy, PyErr, PyObject, PyResult, Python, ToPyObject};
use std::ffi::CString;
use std::sync::{Arc, Mutex};
use wgpu::MaintainBase;

lazy_static! {
    static ref PYTHON_MODULE: Mutex<Option<PyObject>> = Mutex::new(None);
}

fn initialize_python_module(py: Python) -> PyResult<()> {
    let text = std::fs::read_to_string(SHADER_ASSET_PATH).unwrap();
    let module = PyModule::from_code(
        py,
        CString::new(text).unwrap().as_c_str(),
        c"bevy_cuda_script.py",
        c"bevy_cuda_module",
    )?;

    *PYTHON_MODULE.lock().unwrap() = Some(module.into_py(py));
    Ok(())
}

const SHADER_ASSET_PATH: &str = "assets/test.py";

pub struct CudaPlugin;

#[derive(Resource, ExtractResource, Clone)]
pub struct TheImage(Handle<Image>);

fn startup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    asset_loader: Res<AssetServer>,
) {
    let size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };

    Python::with_gil(|py| initialize_python_module(py)).unwrap();

    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Bgra8UnormSrgb,
        RenderAssetUsages::default(),
    );

    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::RENDER_ATTACHMENT;

    let image = images.add(image);
    commands.insert_resource(TheImage(image));
}

impl Plugin for CudaPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<TheImage>::default())
            .add_systems(Startup, startup);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<ViewNodeRunner<CudaNode>>(Core3d, CudaLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    CudaLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.insert_resource(CudaDevice(cudarc::driver::CudaDevice::new(0).unwrap()));
    }
}

#[derive(Resource, Deref, DerefMut)]
pub struct CudaDevice(Arc<cudarc::driver::CudaDevice>);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct CudaLabel;

#[derive(Default)]
struct CudaNode;

impl ViewNode for CudaNode {
    type ViewQuery = (&'static ViewTarget,);

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target,): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let render_queue = world.get_resource::<RenderQueue>().unwrap();
        render_queue.submit(render_context.finish());
        render_context.render_device().poll(Maintain::Wait);

        unsafe {
            render_context
                .render_device()
                .wgpu_device()
                .as_hal::<wgpu::hal::api::Vulkan, _, _>(|device| {
                    let Some(device) = device else {
                        panic!("Failed to get Vulkan device");
                    };

                    let image = view_target
                        .main_texture()
                        .as_hal::<wgpu::hal::api::Vulkan, _, _>(|image| {
                            let Some(image) = image else {
                                panic!("Failed to get Vulkan image");
                            };
                            image.raw_handle()
                        });

                    let instance = device.shared_instance().raw_instance();
                    let physical_device = device.raw_physical_device();
                    let device = device.raw_device();
                    let image_width = view_target.main_texture().width();
                    let image_height = view_target.main_texture().height();

                    let res = mem::with_vk_cuda_buffer(
                        instance,
                        device,
                        physical_device,
                        image,
                        image_width,
                        image_height,
                        vk::Format::R8G8B8A8_SRGB,
                        0,
                        0,
                        |py, cuda_array| {
                            let module_guard = PYTHON_MODULE.lock().unwrap();
                            if let Some(ref module_obj) = *module_guard {
                                let module = module_obj.bind(py).downcast::<PyModule>()?;

                                let res = module.getattr("process_image")?.call1((cuda_array,))?;
                                Ok(())
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                                    "Python module not initialized",
                                ))
                            }
                        },
                    );

                    if let Err(err) = res {
                        eprintln!("CUDA processing failed: {:?}", err);
                    }
                });
        }

        render_context.render_device().poll(Maintain::Wait);

        Ok(())
    }
}
