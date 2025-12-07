use crate::abstract_segment::{AbstractLineSegment, SegType};
use crate::create_paint_array;
use crate::path::{AbstractPath, Paint};
use std::fs;
use usvg::tiny_skia_path::{PathSegment, Point};
use usvg::{Group, Node, Path};

pub fn create_abstract_segment_array(
    abs_segments: &mut Vec<AbstractLineSegment>,
    path: &Path,
    path_idx: usize,
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

pub fn parse_svg()
-> Result<(Vec<AbstractPath>, Vec<AbstractLineSegment>, Vec<Paint>), Box<dyn std::error::Error>> {
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
    // Parse SVG to normal paths
    visit_group(svg_tree.root(), &mut paths);

    // Convert paths to abstract paths, segments, paints
    let mut seg_start_idx = 0usize;
    for (i, path) in paths.iter().enumerate() {
        let seg_count = create_abstract_segment_array(&mut abs_segments, path, i);
        let seg_end_idx = seg_start_idx + seg_count;
        abs_paths.push(AbstractPath {
            seg_start_idx,
            seg_end_idx,
            fill_rule: usvg::FillRule::EvenOdd,
            paint_id: i,
            bounding_box: path.bounding_box(),
        });
        seg_start_idx = seg_end_idx;
        // TODO: For now we have same number of paints as paths
        create_paint_array(&mut paints, path);
    }
    Ok((abs_paths, abs_segments, paints))
}
