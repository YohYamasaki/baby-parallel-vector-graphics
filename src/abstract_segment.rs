use crate::geometry::rect::Rect;
use bytemuck::{Pod, Zeroable};
use usvg::tiny_skia_path::Point;

const EPS: f32 = 1e-6;

#[derive(Debug, PartialEq, Clone)]
pub enum Direction {
    NW,
    NE,
    SW,
    SE,
    Horizontal,
}

impl Direction {
    fn from_u32(int: u32) -> Self {
        match int {
            0 => Direction::NW,
            1 => Direction::NE,
            2 => Direction::SW,
            3 => Direction::SE,
            4 => Direction::Horizontal,
            _ => {
                panic!("Invalid integer passed")
            }
        }
    }
    fn to_u32(&self) -> u32 {
        match self {
            Direction::NW => 0,
            Direction::NE => 1,
            Direction::SW => 2,
            Direction::SE => 3,
            Direction::Horizontal => 4,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum SegType {
    Point,
    Linear,
    Quadratic,
    Cubic,
    Arc,
    Path,
    LastGeom,
    FirstStack,
    Push,
    PopFill,
    PopClip,
    Commit,
    LastStack,
}

impl SegType {
    pub fn to_u32(&self) -> u32 {
        match self {
            SegType::Point => 0,
            SegType::Linear => 1,
            SegType::Quadratic => 2,
            SegType::Cubic => 3,
            SegType::Arc => 4,
            SegType::Path => 5,
            SegType::LastGeom => 6,
            SegType::FirstStack => 7,
            SegType::Push => 8,
            SegType::PopFill => 9,
            SegType::PopClip => 10,
            SegType::Commit => 11,
            SegType::LastStack => 12,
        }
    }
}

impl Direction {
    pub fn to_winding_inc(&self) -> i32 {
        match self {
            Direction::NE | Direction::NW => 1,
            Direction::SE | Direction::SW => -1,
            _ => 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct AbstractLineSegment {
    pub seg_type: u32,
    pub path_idx: u32,
    pub _pad0: [u32; 2],

    pub bbox_ltrb: [f32; 4],

    pub direction: u32,
    // Coefficients for the implicit line equation ax + by + c = 0.
    a: f32,
    b: f32,
    c: f32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl AbstractLineSegment {
    pub fn new(p0: Point, p1: Point, seg_type: SegType, path_id: u32) -> Self {
        let dir = Self::direction_svg(p1.x - p0.x, p1.y - p0.y);
        let bounding_box = Self::line_bbox(&p0, &p1);
        let mut a = p0.y - p1.y;
        let mut b = p1.x - p0.x;
        let mut c = p0.x * p1.y - p1.x * p0.y;

        if a < 0.0 || (a == 0.0 && b < 0.0) {
            a = -a;
            b = -b;
            c = -c;
        }

        AbstractLineSegment {
            seg_type: seg_type.to_u32(),
            direction: dir.to_u32(),
            bbox_ltrb: bounding_box.to_ltrb(),
            a,
            b,
            c,
            _pad0: [0; 2],
            path_idx: path_id,
            x0: p0.x,
            y0: p0.y,
            x1: p1.x,
            y1: p1.y,
        }
    }

    #[inline(always)]
    pub fn eval(&self, x: f32, y: f32) -> f32 {
        self.a * x + self.b * y + self.c
    }

    #[inline(always)]
    pub fn is_left(&self, x: f32, y: f32) -> bool {
        self.eval(x, y) < 0.
    }

    pub fn going_right(&self) -> bool {
        let dir = Direction::from_u32(self.direction);
        match dir {
            Direction::NW => false,
            Direction::NE => true,
            Direction::SW => false,
            Direction::SE => true,
            Direction::Horizontal => true,
        }
    }

    pub fn going_up(&self) -> bool {
        let dir = Direction::from_u32(self.direction);
        match dir {
            Direction::NW => true,
            Direction::NE => true,
            Direction::SW => false,
            Direction::SE => false,
            Direction::Horizontal => false,
        }
    }

    pub fn hit_chull(&self, pt: &Point) -> i32 {
        -1
    }

    /// Returns the x coordinate on the line at the given y.
    fn x_at_y(&self, y0: f32) -> Option<f32> {
        if self.a.abs() < EPS {
            return None;
        }
        Some(-(self.b * y0 + self.c) / self.a)
    }

    /// Returns true if the segment crosses any edge of `bb`.
    pub fn intersect_with_bb(&self, bb: &Rect) -> bool {
        let bounding_box = Rect::from_ltrb_slice(&self.bbox_ltrb);
        if self.is_inside_bb(bb) || bounding_box.unwrap().intersect(&bb).is_none() {
            return false;
        }

        if self.eval(bb.left(), bb.top()) * self.eval(bb.right(), bb.top()) < 0.0 {
            return true; // top
        }
        if self.eval(bb.right(), bb.top()) * self.eval(bb.right(), bb.bottom()) < 0.0 {
            return true; // right
        }
        if self.eval(bb.left(), bb.bottom()) * self.eval(bb.right(), bb.bottom()) < 0.0 {
            return true; // bottom
        }
        if self.eval(bb.left(), bb.top()) * self.eval(bb.left(), bb.bottom()) < 0.0 {
            return true; // left
        }
        false
    }

    pub fn is_inside_bb(&self, bb: &Rect) -> bool {
        bb.left() < self.bbox_ltrb[0]
            && bb.top() < self.bbox_ltrb[1]
            && bb.right() > self.bbox_ltrb[2]
            && bb.bottom() > self.bbox_ltrb[3]
    }

    fn direction_svg(dx: f32, dy: f32) -> Direction {
        if dy.abs() < EPS {
            return Direction::Horizontal;
        }

        match (dx >= 0.0, dy >= 0.0) {
            (true, false) => Direction::NE,  // +x, -y
            (false, false) => Direction::NW, // -x, -y
            (true, true) => Direction::SE,   // +x, +y
            (false, true) => Direction::SW,  // -x, +y
        }
    }

    fn line_bbox(a: &Point, b: &Point) -> Rect {
        let left = a.x.min(b.x);
        let right = a.x.max(b.x);
        let top = a.y.min(b.y);
        let bottom = a.y.max(b.y);
        Rect::from_ltrb(left, top, right, bottom).unwrap()
    }

    pub fn hit_shortcut(&self, cell: &Rect, sample_x: f32, sample_y: f32) -> bool {
        if self.b.abs() < EPS {
            // Ignore if no slope
            return false;
        }
        let x0 = cell.right();
        // y of the segment endpoint with larger x (the shortcut base).
        let y0 = if self.x0 > self.x1 { self.y0 } else { self.y1 };

        if sample_y >= y0 {
            return false;
        }
        sample_x < x0
    }

    pub fn get_shortcut_base(&self) -> [f32; 2] {
        if self.x0 > self.x1 {
            [self.x0, self.y0]
        } else {
            [self.x1, self.y1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static PATH_ID: u32 = 0;

    #[test]
    fn direction_sw() {
        let a = Point { x: 1., y: 0. };
        let b = Point { x: 0., y: 1. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let dir = Direction::from_u32(abs_seg.direction);
        assert_eq!(dir, Direction::SW);
    }

    #[test]
    fn direction_se() {
        let a = Point { x: 0., y: 0. };
        let b = Point { x: 1., y: 1. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let dir = Direction::from_u32(abs_seg.direction);
        assert_eq!(dir, Direction::SE);
    }

    #[test]
    fn direction_nw() {
        let a = Point { x: 1., y: 1. };
        let b = Point { x: 0., y: 0. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let dir = Direction::from_u32(abs_seg.direction);
        assert_eq!(dir, Direction::NW);
    }

    #[test]
    fn direction_ne() {
        let a = Point { x: 0., y: 1. };
        let b = Point { x: 1., y: 0. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let dir = Direction::from_u32(abs_seg.direction);
        assert_eq!(dir, Direction::NE);
    }

    #[test]
    fn bounding_box() {
        let a = Point { x: 2., y: 0. };
        let b = Point { x: 0., y: 3. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        assert_eq!(abs_seg.bbox_ltrb, [0., 0., 2., 3.]);
    }

    #[test]
    fn is_left_pt1() {
        let a = Point { x: 2., y: 0. };
        let b = Point { x: 0., y: 3. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        println!("{:?}", &abs_seg);
        let sample = Point { x: 1., y: 1. };
        assert!(abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_left_pt2() {
        let a = Point { x: 0., y: 3. };
        let b = Point { x: 2., y: 0. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        println!("{:?}", &abs_seg);
        let sample = Point { x: 1., y: 1. };
        assert!(abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_left_pt3() {
        let a = Point { x: 20., y: 80. };
        let b = Point { x: 50., y: 20. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        println!("{:?}", abs_seg);
        let sample = Point { x: 50., y: 50. };
        assert!(!abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_right() {
        let a = Point { x: 0., y: 0. };
        let b = Point { x: 3., y: 2. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let sample = Point { x: 1.51, y: 1. };
        assert!(!abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_in_bb() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 50., y: 80. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let bb = Rect::from_ltrb(0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(abs_seg.is_inside_bb(&bb));
    }

    #[test]
    fn intersect_with_bb() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 40., y: 90. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let bb = Rect::from_ltrb(50.0, 50.0, 100.0, 100.0).unwrap();
        assert!(!abs_seg.intersect_with_bb(&bb));
    }

    #[test]
    fn not_intersect_with_bb_pt1() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 50., y: 80. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let bb = Rect::from_ltrb(0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(!abs_seg.intersect_with_bb(&bb));
    }

    #[test]
    fn not_intersect_with_bb_pt2() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 30., y: 30. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear, PATH_ID);
        let bb = Rect::from_ltrb(50.0, 50.0, 100.0, 100.0).unwrap();
        assert!(!abs_seg.intersect_with_bb(&bb));
    }
}
