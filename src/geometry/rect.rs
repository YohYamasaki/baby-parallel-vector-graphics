fn checked_f32_sub(a: f32, b: f32) -> Option<f32> {
    debug_assert!(a.is_finite());
    debug_assert!(b.is_finite());

    let n = a as f64 - b as f64;
    if n > f32::MIN as f64 && n < f32::MAX as f64 {
        Some(n as f32)
    } else {
        None
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Rect {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl Rect {
    pub(crate) fn from_ltrb_slice(s: &[f32; 4]) -> Option<Self> {
        Rect::from_ltrb(s[0], s[1], s[2], s[3])
    }
}

impl Rect {
    pub fn from_ltrb(left: f32, top: f32, right: f32, bottom: f32) -> Option<Self> {
        if left <= right && top <= bottom {
            checked_f32_sub(right, left)?;
            checked_f32_sub(bottom, top)?;
            Some(Rect {
                left,
                top,
                right,
                bottom,
            })
        } else {
            None
        }
    }

    pub fn to_ltrb(&self) -> [f32; 4] {
        [self.left, self.top, self.right, self.bottom]
    }

    /// Returns the left edge.
    pub fn left(&self) -> f32 {
        self.left
    }

    /// Returns the top edge.
    pub fn top(&self) -> f32 {
        self.top
    }

    /// Returns the right edge.
    pub fn right(&self) -> f32 {
        self.right
    }

    /// Returns the bottom edge.
    pub fn bottom(&self) -> f32 {
        self.bottom
    }

    /// Returns rect's X position.
    pub fn x(&self) -> f32 {
        self.left
    }

    /// Returns rect's Y position.
    pub fn y(&self) -> f32 {
        self.top
    }

    pub fn width(&self) -> f32 {
        self.right - self.left
    }

    pub fn height(&self) -> f32 {
        self.bottom - self.top
    }

    pub fn mid_point(&self) -> [f32; 2] {
        [
            self.left + (self.width() / 2.0),
            self.top() + (self.height() / 2.0),
        ]
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let left = self.x().max(other.x());
        let top = self.y().max(other.y());

        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());

        Rect::from_ltrb(left, top, right, bottom)
    }
}
