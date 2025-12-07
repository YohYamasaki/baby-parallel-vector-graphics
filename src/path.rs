use usvg::FillRule;
use usvg::Rect;

#[derive(Debug)]
pub struct AbstractPath {
    pub seg_start_idx: usize,
    pub seg_end_idx: usize,
    pub fill_rule: FillRule,
    pub paint_id: usize,
    pub bounding_box: Rect,
}

#[derive(Debug)]
pub enum Paint {
    SolidColor { rgba: [u8; 4] },
}
