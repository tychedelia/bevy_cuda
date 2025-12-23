use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy_cuda::{compile_ptx, CudaBuffer, CudaContext, CudaModule, CudaPlugin, LaunchConfig, PushKernelArg, TextureUses};
use std::sync::Arc;

const ANIMATE_KERNEL: &str = r#"
extern "C" __global__ void animate_colors(unsigned char* data, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 4;

        float fx = (float)x / (float)width;
        float fy = (float)y / (float)height;

        float r = 0.5f + 0.5f * sinf(fx * 6.28f + time);
        float g = 0.5f + 0.5f * sinf(fy * 6.28f + time * 1.3f);
        float b = 0.5f + 0.5f * sinf((fx + fy) * 3.14f + time * 0.7f);

        data[idx + 0] = (unsigned char)(r * 255.0f);
        data[idx + 1] = (unsigned char)(g * 255.0f);
        data[idx + 2] = (unsigned char)(b * 255.0f);
        data[idx + 3] = 255;
    }
}
"#;

const TEX_WIDTH: u32 = 256;
const TEX_HEIGHT: u32 = 256;

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .add_plugins(CudaPlugin::default())
        .add_plugins(ExtractResourcePlugin::<CudaTexture>::default())
        .add_plugins(ExtractResourcePlugin::<AnimationTime>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate_cube, update_time));

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(RenderStartup, setup_cuda_resources);
        render_app.add_systems(
            Render,
            cuda_animate_texture.in_set(RenderSystems::PrepareAssets),
        );
    }

    app.run();
}

#[derive(Component)]
struct RotatingCube;

#[derive(Resource, Clone, ExtractResource)]
struct CudaTexture(Handle<Image>);

#[derive(Resource, Clone, ExtractResource, Default)]
struct AnimationTime(f32);

#[derive(Resource)]
struct CudaResources {
    buffer: CudaBuffer,
    module: Arc<CudaModule>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let mut texture = Image::new_fill(
        Extent3d {
            width: TEX_WIDTH,
            height: TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[128, 128, 128, 255],
        TextureFormat::Rgba8Unorm,
        default(),
    );
    // we need COPY_SRC and COPY_DST so that we can roundtrip through cuda
    texture.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::COPY_DST;

    let texture_handle = images.add(texture);
    commands.insert_resource(CudaTexture(texture_handle.clone()));
    commands.insert_resource(AnimationTime(0.0));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(texture_handle),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        RotatingCube,
    ));

    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn rotate_cube(time: Res<Time>, mut query: Query<&mut Transform, With<RotatingCube>>) {
    for mut transform in &mut query {
        transform.rotate_y(time.delta_secs() * 0.5);
    }
}

fn update_time(time: Res<Time>, mut anim_time: ResMut<AnimationTime>) {
    anim_time.0 += time.delta_secs();
}

fn setup_cuda_resources(mut commands: Commands, cuda: Res<CudaContext>) {
    let buffer_size = (TEX_WIDTH * TEX_HEIGHT * 4) as u64;

    let buffer = match cuda.create_buffer(buffer_size) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to create CUDA buffer: {:?}", e);
            return;
        }
    };

    let module = match compile_ptx(ANIMATE_KERNEL) {
        Ok(ptx) => match cuda.load_module(ptx) {
            Ok(m) => m,
            Err(e) => {
                error!("Failed to load module: {:?}", e);
                return;
            }
        },
        Err(e) => {
            error!("Failed to compile PTX: {:?}", e);
            return;
        }
    };

    info!("CUDA resources initialized");
    commands.insert_resource(CudaResources { buffer, module });
}

fn cuda_animate_texture(
    cuda_texture: Option<Res<CudaTexture>>,
    anim_time: Option<Res<AnimationTime>>,
    cuda: Res<CudaContext>,
    resources: Option<Res<CudaResources>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(cuda_texture) = cuda_texture else { return };
    let Some(anim_time) = anim_time else { return };
    let Some(resources) = resources else { return };
    let Some(gpu_image) = gpu_images.get(&cuda_texture.0) else { return };

    let device = render_device.wgpu_device();
    let queue: &wgpu::Queue = &render_queue;
    let width = gpu_image.size.width;
    let height = gpu_image.size.height;

    let func = match resources.module.get_function("animate_colors") {
        Ok(f) => f,
        Err(e) => {
            error_once!("get_function error: {:?}", e);
            return;
        }
    };

    let stream = cuda.default_stream();
    let config = LaunchConfig {
        grid_dim: ((width + 15) / 16, (height + 15) / 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let ptr = resources.buffer.ptr();
    let w = width as i32;
    let h = height as i32;
    let t = anim_time.0;

    unsafe {
        if let Err(e) = stream
            .launch_builder(&func)
            .arg(&ptr)
            .arg(&w)
            .arg(&h)
            .arg(&t)
            .launch(config)
        {
            error_once!("kernel launch error: {:?}", e);
            return;
        }
    }
    stream.synchronize().ok();

    if let Err(e) = cuda.copy_buffer_to_texture(
        device,
        queue,
        &resources.buffer,
        &gpu_image.texture,
        width,
        height,
        gpu_image.texture_format,
        TextureUses::RESOURCE,
    ) {
        error_once!("copy_buffer_to_texture error: {:?}", e);
    }
}
