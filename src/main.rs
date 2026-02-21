mod abstract_segment;
mod cell_entry;
mod geometry;
mod gpu;
mod path;
mod png_writer;
mod quad_tree;
mod render;
mod svg_parser;

use crate::cell_entry::init_root_cell_entries;
use crate::geometry::rect::Rect;
use crate::gpu::quad_tree::build_quadtree;
use crate::gpu::render::{build_path_paints, ComputeRenderer};
use crate::path::Paint;
use crate::png_writer::save_png_rgba8;
use crate::quad_tree::QuadTree;
use crate::render::render;
use crate::svg_parser::parse_svg;
use std::sync::Arc;
use usvg::Path;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::window::Window;

fn main() -> anyhow::Result<()> {
    let (abs_paths, abs_segments, paints) = parse_svg()?;

    let root_bounds = Rect::from_ltrb(0.0, 0.0, 1000.0, 1000.0).unwrap();
    let root_entries = init_root_cell_entries(&abs_segments);
    let (metadata, cell_entry) = build_quadtree(root_bounds, root_entries, 4, 1, &abs_segments)?;
    let path_paints = build_path_paints(&abs_paths, &paints);

    // Rendering on GPU, compute to offscreen texture + surface blit + PNG readback
    let size = 1000u32;
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        event_loop.create_window(
            Window::default_attributes()
                .with_title("baby-parallel-vector-graphics")
                .with_visible(false)
                .with_inner_size(PhysicalSize::new(size, size)),
        )?,
    );

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone())?;
    let renderer = pollster::block_on(ComputeRenderer::new(&instance, &surface, size, size))?;
    let gpu_pixels = renderer.render_to_rgba(
        &surface,
        &metadata,
        &cell_entry,
        &abs_segments,
        &path_paints,
    )?;
    save_png_rgba8("output/test_gpu.png", size, size, &gpu_pixels);

    // Rendering on CPU for reference
    let render_tree = QuadTree::new(&abs_segments, root_bounds, 4, 1)?;
    let mut cpu_pixels = [0u8; 4_000_000];
    render(
        &render_tree,
        &abs_segments,
        &abs_paths,
        &paints,
        &mut cpu_pixels,
        size,
        size,
    );
    save_png_rgba8("output/test_cpu.png", size, size, &cpu_pixels);
    Ok(())
}

fn create_paint_array(paints: &mut Vec<Paint>, path: &Path) {
    let fill = path.fill().unwrap().paint();
    if let usvg::Paint::Color(c) = fill {
        paints.push(Paint::SolidColor {
            rgba: [c.red, c.green, c.blue, 255],
        });
    }
}
