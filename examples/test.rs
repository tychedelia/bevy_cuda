use std::env;
use std::path::PathBuf;
use bevy::prelude::*;
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::RenderPlugin;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy_cuda::CudaPlugin;
use pyo3::{ffi, PyErr, PyResult, Python};
use pyo3::ffi::Py_SetPythonHome;
use pyo3::prelude::PyAnyMethods;

fn main() -> PyResult<()> {
    // Get venv path from environment variable
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

    // Build site-packages path based on platform
    let site_packages = if cfg!(windows) {
        format!("{}\\Lib\\site-packages", venv_path)
    } else {
        format!("{}/lib/python3.11/site-packages", venv_path)
    };

    // Make sure the path exists
    if !PathBuf::from(&site_packages).exists() {
        eprintln!("Site packages directory not found: {}", site_packages);
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Site packages directory not found: {}", site_packages),
        ));
    }

    println!("Adding to Python path: {}", site_packages);

    // Initialize Python
    pyo3::prepare_freethreaded_python();

    App::new()
        .add_plugins(DefaultPlugins.set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                // WARN this is a native only feature. It will not work with webgl or webgpu
                features: WgpuFeatures::VULKAN_EXTERNAL_MEMORY_WIN32,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(CudaPlugin)
        .add_systems(Startup, setup)
        .run();

    Ok(())
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // circular base
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
