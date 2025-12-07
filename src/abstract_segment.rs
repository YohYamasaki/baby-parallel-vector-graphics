use usvg::{tiny_skia_path::Point, Rect};

const EPS: f32 = 1e-6;

#[derive(Debug, PartialEq, Clone)]
pub enum Direction {
    NW,
    NE,
    SW,
    SE,
    Horizontal, // TODO: split to W/E?
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
impl Direction {
    pub fn to_winding_inc(&self) -> i32 {
        match self {
            Direction::NE | Direction::NW => 1,
            Direction::SE | Direction::SW => -1,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImplicitLine {
    // coefficients for implicit function: ax+by+c
    a: f32,
    b: f32,
    c: f32,
}

impl ImplicitLine {
    pub fn new(p0: &Point, p1: &Point) -> Self {
        let mut a = p0.y - p1.y;
        let mut b = p1.x - p0.x;
        let mut c = p0.x * p1.y - p1.x * p0.y;

        if a < 0.0 || (a == 0.0 && b < 0.0) {
            a = -a;
            b = -b;
            c = -c;
        }
        Self { a, b, c }
    }

    #[inline(always)]
    fn eval(&self, x: f32, y: f32) -> f32 {
        self.a * x + self.b * y + self.c
    }
}

#[derive(Debug, Clone)]
pub struct AbstractLineSegment {
    pub seg_type: SegType,
    pub path_idx: usize,
    pub bounding_box: Rect,
    pub direction: Direction,
    implicit_line: ImplicitLine,
    pub p0: Point, // start point
    pub p1: Point, // end point
}

impl AbstractLineSegment {
    pub fn new(p0: Point, p1: Point, seg_type: SegType, path_id: usize) -> Self {
        let direction = Self::direction_svg(p1.x - p0.x, p1.y - p0.y);
        let bounding_box = Self::line_bbox(&p0, &p1);

        let mut a = p0.y - p1.y;
        let mut b = p1.x - p0.x;
        let mut c = p0.x * p1.y - p1.x * p0.y;

        AbstractLineSegment {
            seg_type,
            direction,
            bounding_box,
            implicit_line: ImplicitLine::new(&p0, &p1),
            path_idx: path_id,
            p0,
            p1,
        }
    }

    #[inline(always)]
    pub(crate) fn eval(&self, x: f32, y: f32) -> f32 {
        self.implicit_line.eval(x, y)
    }

    #[inline(always)]
    pub fn is_left(&self, x: f32, y: f32) -> bool {
        self.eval(x, y) < 0.
    }

    pub fn going_right(&self) -> bool {
        match self.direction {
            Direction::NW => false,
            Direction::NE => true,
            Direction::SW => false,
            Direction::SE => true,
            Direction::Horizontal => true,
        }
    }

    pub fn going_up(&self) -> bool {
        match self.direction {
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

    /// Returns x position of the given y.
    fn x_at_y(&self, y0: f32) -> Option<f32> {
        let il = &self.implicit_line;
        if il.a.abs() < EPS {
            return None; // Horizontal
        }
        Some(-(il.b * y0 + il.c) / il.a)
    }

    /// Check if the segment intersects with one of the boundaries of the given bounding box.
    pub fn intersect_with_bb(&self, bb: &Rect) -> bool {
        if self.is_inside_bb(bb) || self.bounding_box.intersect(&bb).is_none() {
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
        let seg_bb = &self.bounding_box;
        bb.top() < seg_bb.top()
            && bb.right() > seg_bb.right()
            && bb.bottom() > seg_bb.bottom()
            && bb.left() < seg_bb.left()
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
        if self.implicit_line.b.abs() < EPS {
            // Ignore if no slope
            return false;
        }
        let x0 = cell.right();
        // Use y position of the right end of the segment
        let y0 = if self.p0.x > self.p1.x {
            self.p0.y
        } else {
            self.p1.y
        };

        if sample_y >= y0 {
            return false;
        }
        if sample_x < x0 { true } else { false }
    }

    pub fn get_shortcut_base(&self) -> &Point {
        if self.p0.x > self.p1.x {
            &self.p0
        } else {
            &self.p1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static PATH_ID: usize = 0;

    #[test]
    fn direction_sw() {
        let a = Point { x: 1., y: 0. };
        let b = Point { x: 0., y: 1. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        assert_eq!(abs_seg.direction, Direction::SW);
    }

    #[test]
    fn direction_se() {
        let a = Point { x: 0., y: 0. };
        let b = Point { x: 1., y: 1. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        assert_eq!(abs_seg.direction, Direction::SE);
    }

    #[test]
    fn direction_nw() {
        let a = Point { x: 1., y: 1. };
        let b = Point { x: 0., y: 0. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        assert_eq!(abs_seg.direction, Direction::NW);
    }

    #[test]
    fn direction_ne() {
        let a = Point { x: 0., y: 1. };
        let b = Point { x: 1., y: 0. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        assert_eq!(abs_seg.direction, Direction::NE);
    }

    #[test]
    fn bounding_box() {
        let a = Point { x: 2., y: 0. };
        let b = Point { x: 0., y: 3. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        let expected = Rect::from_ltrb(0., 0., 2., 3.).unwrap();
        assert_eq!(abs_seg.bounding_box, expected);
    }

    #[test]
    fn is_left_pt1() {
        let a = Point { x: 2., y: 0. };
        let b = Point { x: 0., y: 3. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        println!("{:?}", &abs_seg);
        let sample = Point { x: 1., y: 1. };
        assert!(abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_left_pt2() {
        let a = Point { x: 0., y: 3. };
        let b = Point { x: 2., y: 0. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        println!("{:?}", &abs_seg);
        let sample = Point { x: 1., y: 1. };
        assert!(abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_left_pt3() {
        let a = Point { x: 20., y: 80. };
        let b = Point { x: 50., y: 20. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        println!("{:?}", abs_seg);
        let sample = Point { x: 50., y: 50. };
        assert!(!abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_right() {
        let a = Point { x: 0., y: 0. };
        let b = Point { x: 3., y: 2. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        let sample = Point { x: 1.51, y: 1. };
        assert!(!abs_seg.is_left(sample.x, sample.y));
    }

    #[test]
    fn is_in_bb() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 50., y: 80. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        let bb = Rect::from_ltrb(0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(abs_seg.is_inside_bb(&bb));
    }

    #[test]
    fn intersect_with_bb() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 40., y: 90. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        let bb = Rect::from_ltrb(50.0, 50.0, 100.0, 100.0).unwrap();
        assert!(!abs_seg.intersect_with_bb(&bb));
    }

    #[test]
    fn not_intersect_with_bb_pt1() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 50., y: 80. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        let bb = Rect::from_ltrb(0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(!abs_seg.intersect_with_bb(&bb));
    }

    #[test]
    fn not_intersect_with_bb_pt2() {
        let a = Point { x: 20., y: 20. };
        let b = Point { x: 30., y: 30. };
        let abs_seg = AbstractLineSegment::new(a, b, SegType::Linear ,PATH_ID);
        let bb = Rect::from_ltrb(50.0, 50.0, 100.0, 100.0).unwrap();
        assert!(!abs_seg.intersect_with_bb(&bb));
    }
}
