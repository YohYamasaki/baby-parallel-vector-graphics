use crate::abstract_segment::{AbstractLineSegment, SegType};
use crate::create_paint_array;
use crate::path::{AbstractPath, Paint};
use anyhow::Context;
use crate::geometry::rect::Rect;
use std::fs;
use usvg::tiny_skia_path::{PathSegment, Point};
use usvg::{Group, Node, Path};

pub fn create_abstract_segment_array(
    abs_segments: &mut Vec<AbstractLineSegment>,
    path: &Path,
    path_idx: u32,
) -> usize {
    let mut start: Option<Point> = None;
    let mut curr: Option<Point> = None;
    let mut seg_count = 0usize;

    for segment in path.data().segments() {
        match segment {
            PathSegment::MoveTo(point) => {
                start = Some(point);
                curr = Some(point);
            }
            PathSegment::LineTo(point) => {
                let a = curr.expect("There should be a point before");
                curr = Some(point);
                abs_segments.push(AbstractLineSegment::new(
                    a,
                    point,
                    SegType::Linear,
                    path_idx,
                ));
                seg_count += 1;
            }
            PathSegment::QuadTo(_, _) => todo!(),
            PathSegment::CubicTo(_, _, _) => todo!(),
            PathSegment::Close => {
                let a = curr.expect("There should be at least one point");
                let b = start.expect("There should be at least one point");
                abs_segments.push(AbstractLineSegment::new(a, b, SegType::Linear, path_idx));
                seg_count += 1;
            }
        }
    }
    seg_count
}

pub fn visit_group(g: &Group, paths: &mut Vec<Path>) {
    for node in g.children() {
        match node {
            Node::Path(p) => {
                paths.push(*p.clone());
            }
            Node::Group(child) => visit_group(child, paths),
            Node::Image(_) => {}
            Node::Text(_) => {}
        }
    }
}

pub struct ParsedSvg {
    pub abs_paths: Vec<AbstractPath>,
    pub abs_segments: Vec<AbstractLineSegment>,
    pub paints: Vec<Paint>,
    pub width: u32,
    pub height: u32,
}

pub fn parse_svg() -> anyhow::Result<ParsedSvg> {
    let mut paths: Vec<Path> = vec![];
    let mut abs_paths: Vec<AbstractPath> = vec![];
    let mut abs_segments: Vec<AbstractLineSegment> = vec![];
    let mut paints: Vec<Paint> = vec![];

    let svg_path = format!(
        "{}/sample_svg/simple_polygons.svg",
        env!("CARGO_MANIFEST_DIR")
    );
    let svg: String = fs::read_to_string(svg_path)?;
    let opt = usvg::Options::default();
    let svg_tree = usvg::Tree::from_str(&svg, &opt)?;
    visit_group(svg_tree.root(), &mut paths);

    let mut seg_start_idx = 0usize;
    for (i, path) in paths.iter().enumerate() {
        let seg_count = create_abstract_segment_array(&mut abs_segments, path, i as u32);
        let seg_end_idx = seg_start_idx + seg_count;
        let bb = path.bounding_box();
        abs_paths.push(AbstractPath {
            seg_start_idx,
            seg_end_idx,
            fill_rule: usvg::FillRule::EvenOdd,
            paint_id: i,
            bounding_box: Rect::from_ltrb(bb.left(), bb.top(), bb.right(), bb.bottom()).unwrap(),
        });
        seg_start_idx = seg_end_idx;
        create_paint_array(&mut paints, path);
    }
    let svg_size = svg_tree.size();
    let width = svg_size.width().ceil().max(1.0) as u32;
    let height = svg_size.height().ceil().max(1.0) as u32;
    let _ = Rect::from_ltrb(0.0, 0.0, width as f32, height as f32)
        .context("Invalid parsed SVG size")?;

    Ok(ParsedSvg {
        abs_paths,
        abs_segments,
        paints,
        width,
        height,
    })
}
