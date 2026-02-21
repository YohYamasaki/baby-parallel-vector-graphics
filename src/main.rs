mod abstract_segment;
mod cell_entry;
mod geometry;
mod gpu;
mod path;
mod png_writer;
mod quad_tree;
mod render;
mod svg_parser;

use crate::abstract_segment::{AbstractLineSegment, SegType};
use crate::cell_entry::init_root_cell_entries;
use crate::geometry::rect::Rect;
use crate::gpu::quad_tree::build_quadtree;
use crate::gpu::render::{build_path_paints, ComputeRenderer};
use crate::path::AbstractPath;
use crate::path::Paint;
use crate::png_writer::save_png_rgba8;
use crate::quad_tree::QuadTree;
use crate::render::render;
use crate::svg_parser::{parse_svg, ParsedSvg};
use std::sync::Arc;
use usvg::tiny_skia_path::Point;
use usvg::Path;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::window::Window;

const OUTPUT_WIDTH_OVERRIDE: Option<u32> = None;
const OUTPUT_HEIGHT_OVERRIDE: Option<u32> = None;

fn main() -> anyhow::Result<()> {
    let mut parsed = parse_svg()?;
    let render_width = OUTPUT_WIDTH_OVERRIDE.unwrap_or(parsed.width).max(1);
    let render_height = OUTPUT_HEIGHT_OVERRIDE.unwrap_or(parsed.height).max(1);

    if render_width != parsed.width || render_height != parsed.height {
        let sx = render_width as f32 / parsed.width as f32;
        let sy = render_height as f32 / parsed.height as f32;
        scale_geometry(&mut parsed.abs_paths, &mut parsed.abs_segments, sx, sy)?;
    }
    let ParsedSvg {
        abs_paths,
        abs_segments,
        paints,
        ..
    } = parsed;

    let root_bounds = Rect::from_ltrb(0.0, 0.0, render_width as f32, render_height as f32).unwrap();
    let root_entries = init_root_cell_entries(&abs_segments);
    let (metadata, cell_entry) = build_quadtree(root_bounds, root_entries, 4, 1, &abs_segments)?;
    let path_paints = build_path_paints(&abs_paths, &paints);

    // Rendering on GPU, compute to offscreen texture + surface blit + PNG readback
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        event_loop.create_window(
            Window::default_attributes()
                .with_title("baby-parallel-vector-graphics")
                .with_visible(false)
                .with_inner_size(PhysicalSize::new(render_width, render_height)),
        )?,
    );

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone())?;
    let renderer = pollster::block_on(ComputeRenderer::new(
        &instance,
        &surface,
        render_width,
        render_height,
    ))?;
    let gpu_pixels = renderer.render_to_rgba(
        &surface,
        &metadata,
        &cell_entry,
        &abs_segments,
        &path_paints,
    )?;
    save_png_rgba8(
        "output/test_gpu.png",
        render_width,
        render_height,
        &gpu_pixels,
    );

    // Rendering on CPU for reference
    let render_tree = QuadTree::new(&abs_segments, root_bounds, 4, 1)?;
    let mut cpu_pixels = vec![0u8; (render_width as usize) * (render_height as usize) * 4];
    render(
        &render_tree,
        &abs_segments,
        &abs_paths,
        &paints,
        &mut cpu_pixels,
        render_width,
        render_height,
    );
    save_png_rgba8(
        "output/test_cpu.png",
        render_width,
        render_height,
        &cpu_pixels,
    );
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

fn scale_geometry(
    abs_paths: &mut [AbstractPath],
    abs_segments: &mut [AbstractLineSegment],
    sx: f32,
    sy: f32,
) -> anyhow::Result<()> {
    for path in abs_paths {
        let [l, t, r, b] = path.bounding_box.to_ltrb();
        path.bounding_box = Rect::from_ltrb(l * sx, t * sy, r * sx, b * sy)
            .ok_or_else(|| anyhow::anyhow!("Invalid path bounding box after scaling"))?;
    }
    for seg in abs_segments {
        let p0 = Point {
            x: seg.x0 * sx,
            y: seg.y0 * sy,
        };
        let p1 = Point {
            x: seg.x1 * sx,
            y: seg.y1 * sy,
        };
        *seg = AbstractLineSegment::new(p0, p1, SegType::Linear, seg.path_idx);
    }
    Ok(())
}
