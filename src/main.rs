mod abstract_segment;
mod cell_entry;
mod cpu_renderer;
mod geometry;
mod gpu;
mod path;
mod png_writer;
mod quad_tree;
mod svg_parser;

use crate::cpu_renderer::render_quadtree_by_node_array;
use crate::geometry::rect::Rect;
use crate::path::Paint;
use crate::png_writer::save_png_rgba8;
use crate::quad_tree::QuadTree;
use crate::svg_parser::parse_svg;
use usvg::Path;

fn main() -> anyhow::Result<()> {
    let (abs_paths, abs_segments, paints) = parse_svg()?;

    // Make vectors immutable
    let abs_paths = abs_paths;
    let paints = paints;

    let root_bounds = Rect::from_ltrb(0.0, 0.0, 1000.0, 1000.0).unwrap();
    let render_tree = QuadTree::new(&abs_segments, root_bounds, 4, 1)?;

    // Debug rendering
    let mut pixels = [0u8; 4000000];
    let size = 1000;
    render_quadtree_by_node_array(
        &render_tree,
        &abs_segments,
        &abs_paths,
        &paints,
        &mut pixels,
        size,
        size,
    );

    save_png_rgba8("output/test.png", size, size, &pixels);
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
