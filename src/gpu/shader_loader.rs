use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

const COMMON: &str = include_str!("common.wgsl");
const SPLIT_HELPERS: &str = include_str!("split_helpers.wgsl");

/// Load a shader by concatenating shared includes with a main shader source.
pub fn load_shader(device: &Device, label: &str, includes: &[&str], main_source: &str) -> ShaderModule {
    let mut combined = String::new();
    for include in includes {
        combined.push_str(include);
        combined.push('\n');
    }
    combined.push_str(main_source);
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(combined.into()),
    })
}

/// Load a shader that only needs common.wgsl.
pub fn load_with_common(device: &Device, label: &str, main_source: &str) -> ShaderModule {
    load_shader(device, label, &[COMMON], main_source)
}

/// Load a shader that needs common.wgsl + split_helpers.wgsl.
pub fn load_with_split_helpers(device: &Device, label: &str, main_source: &str) -> ShaderModule {
    load_shader(device, label, &[COMMON, SPLIT_HELPERS], main_source)
}
