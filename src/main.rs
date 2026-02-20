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
use crate::gpu::render::debug_cpu_render;
use crate::path::Paint;
use crate::png_writer::save_png_rgba8;
use crate::quad_tree::QuadTree;
use crate::render::render;
use crate::svg_parser::parse_svg;
use usvg::Path;

fn main() -> anyhow::Result<()> {
    let (abs_paths, abs_segments, paints) = parse_svg()?;

    // Make vectors immutable
    let abs_paths = abs_paths;
    let paints = paints;

    let root_bounds = Rect::from_ltrb(0.0, 0.0, 1000.0, 1000.0).unwrap();
    let root_entries = init_root_cell_entries(&abs_segments);
    let (metadata, cell_entry) = build_quadtree(root_bounds, root_entries, 4, 0, &abs_segments)?;
    // ----- GPU result debug rendering (CPU rasterization of GPU subdivision output) -----
    let mut gpu_pixels = [0u8; 4000000];
    let size = 1000;
    debug_cpu_render(
        &metadata,
        &cell_entry,
        &abs_segments,
        &abs_paths,
        &paints,
        &mut gpu_pixels,
        size,
        size,
    );
    save_png_rgba8("output/test_gpu.png", size, size, &gpu_pixels);

    // ----- CPU debug rendering -----
    let render_tree = QuadTree::new(&abs_segments, root_bounds, 2, 1)?;
    // Debug rendering
    let mut cpu_pixels = [0u8; 4000000];
    let size = 1000;
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
