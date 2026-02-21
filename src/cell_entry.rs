use crate::abstract_segment::AbstractLineSegment;
use crate::geometry::rect::Rect;
use bytemuck::{Pod, Zeroable};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use usvg::tiny_skia_path::Point;

const NONE_U32: u32 = 0xFFFF_FFFF;
const BOTTOM_LEFT: u32 = 2;
const BOTTOM_RIGHT: u32 = 3;
const TOP_LEFT: u32 = 0;
const TOP_RIGHT: u32 = 1;
pub type EntryFlags = u32;
pub const EMPTY: EntryFlags = 0;
pub const ABSTRACT: EntryFlags = 1 << 0;
pub const WINDING_INCREMENT: EntryFlags = 1 << 3;

static NEXT_CELL_UNIQUE_ID: AtomicU32 = AtomicU32::new(0);

pub type CellId = u32;

fn half_open_eval(seg: &AbstractLineSegment, sample: &Point) -> i32 {
    let [left, top, right, bottom] = seg.bbox_ltrb;

    // Outside the segment's vertical bbox: use a clipped constant sign.
    if sample.y > bottom || sample.y <= top {
        if !(left <= sample.x && sample.x < right) {
            return 0;
        }

        let same_dir = seg.going_right() == seg.going_up();

        // Above or below decides which sign to return.
        // This early return is for avoiding running area evaluation by the implicit function of the segment,
        // but clipping the value
        return if sample.y <= top {
            if same_dir { -1 } else { 1 }
        } else {
            if same_dir { 1 } else { -1 }
        };
    }

    // Within vertical range: classify by x position relative to bbox.
    if sample.x >= right {
        return 1;
    }
    if sample.x < left {
        return -1;
    }

    // Inside bbox: try convex hull, then fall back to implicit evaluation.
    let check = seg.hit_chull(sample);
    if check != -1 {
        return if check == 1 { -1 } else { 1 };
    }

    // Fallback to implicit evaluation sign.
    if seg.eval(sample.x, sample.y) < 0. {
        -1
    } else {
        1
    }
}

struct EdgeIntersectionInfo {
    /*     TL ---10 --- T ---11 --- TR -- 14 -- TI
           |            |           |
           7            8           9
           |            |           |
           L  --- 5 --- C --- 6 --- R  -- 13 -- I
           |            |           |
           2            3           4
           |            |           |
           BL --- 0 --- B --- 1 --- BR -- 12 -- BI

           11 and 14 -> 15
            6 and 13 -> 16
            1 and 12 -> 17
    */
    cross0: bool,
    cross1: bool,
    cross2: bool,
    cross3: bool,

    cross4: bool,
    cross5: bool,
    cross6: bool,
    cross7: bool,

    cross8: bool,
    cross9: bool,
    cross10: bool,
    cross11: bool,

    cross12: bool,
    cross13: bool,
    cross14: bool,
    cross15: bool,

    cross16: bool,
    cross17: bool,
}

impl EdgeIntersectionInfo {
    pub fn new(seg: &AbstractLineSegment, parent_bound: &Rect, mid_point: &Point) -> Self {
        // Extend a ray far beyond the right boundary for winding number evaluation.
        let far_x = parent_bound.right() + (parent_bound.width() + 1.0) * 1024.0;
        let sign_l = half_open_eval(
            &seg,
            &Point {
                x: parent_bound.left(),
                y: mid_point.y,
            },
        );
        let sign_c = half_open_eval(
            &seg,
            &Point {
                x: mid_point.x,
                y: mid_point.y,
            },
        );
        let sign_r = half_open_eval(
            &seg,
            &Point {
                x: parent_bound.right(),
                y: mid_point.y,
            },
        );
        let sign_i = half_open_eval(
            &seg,
            &Point {
                x: far_x,
                y: mid_point.y,
            },
        );
        let sign_bl = half_open_eval(
            &seg,
            &Point {
                x: parent_bound.left(),
                y: parent_bound.bottom(),
            },
        );
        let sign_b = half_open_eval(
            &seg,
            &Point {
                x: mid_point.x,
                y: parent_bound.bottom(),
            },
        );
        let sign_br = half_open_eval(
            &seg,
            &Point {
                x: parent_bound.right(),
                y: parent_bound.bottom(),
            },
        );
        let sign_bi = half_open_eval(
            &seg,
            &Point {
                x: far_x,
                y: parent_bound.bottom(),
            },
        );
        let sign_tl = half_open_eval(
            &seg,
            &Point {
                x: parent_bound.left(),
                y: parent_bound.top(),
            },
        );
        let sign_t = half_open_eval(
            &seg,
            &Point {
                x: mid_point.x,
                y: parent_bound.top(),
            },
        );
        let sign_tr = half_open_eval(
            &seg,
            &Point {
                x: parent_bound.right(),
                y: parent_bound.top(),
            },
        );
        let sign_ti = half_open_eval(
            &seg,
            &Point {
                x: far_x,
                y: parent_bound.top(),
            },
        );
        Self {
            cross0: sign_bl * sign_b < 0,
            cross1: sign_b * sign_br < 0,
            cross2: sign_bl * sign_l < 0,
            cross3: sign_b * sign_c < 0,
            cross4: sign_br * sign_r < 0,
            cross5: sign_l * sign_c < 0,
            cross6: sign_c * sign_r < 0,
            cross7: sign_tl * sign_l < 0,
            cross8: sign_t * sign_c < 0,
            cross9: sign_tr * sign_r < 0,
            cross10: sign_tl * sign_t < 0,
            cross11: sign_t * sign_tr < 0,
            cross12: sign_br * sign_bi < 0,
            cross13: sign_r * sign_i < 0,
            cross14: sign_tr * sign_ti < 0,
            cross15: sign_t * sign_ti < 0,
            cross16: sign_c * sign_i < 0,
            cross17: sign_b * sign_bi < 0,
        }
    }
}

#[inline(always)]
pub const fn flag(x: u32, offset: u32) -> u32 {
    (1u32 << x) << offset
}

pub const fn fill(x: u32) -> u32 {
    flag(x, 0)
}

#[inline]
pub const fn has_fill(split_info: u32, cell: u32) -> bool {
    (split_info & fill(cell)) != 0
}

pub const FILL_MASK: u32 =
    fill(BOTTOM_LEFT) | fill(BOTTOM_RIGHT) | fill(TOP_LEFT) | fill(TOP_RIGHT);

#[inline]
pub const fn has_any_fill(split_info: u32) -> bool {
    (split_info & FILL_MASK) != 0
}
pub const fn up(x: u32) -> u32 {
    flag(x, 4)
}

#[inline]
pub const fn has_up(split_info: u32, cell: u32) -> bool {
    (split_info & up(cell)) != 0
}

pub const fn down(x: u32) -> u32 {
    flag(x, 8)
}

#[inline]
pub const fn has_down(split_info: u32, cell: u32) -> bool {
    (split_info & down(cell)) != 0
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct SplitData {
    winding: [i32; 4],
    split_info: u32,
    _pad: [u32; 3],
}

impl SplitData {
    pub fn new(
        seg: &AbstractLineSegment,
        shortcut: i32,
        einfo: &EdgeIntersectionInfo,
        bound: &Rect,
        mid_point: &Point,
    ) -> Self {
        let going_up = if seg.y0 > seg.y1 { 1 } else { -1 };
        let going_right = if seg.x0 < seg.x1 { 1 } else { -1 };
        let mut split_info = 0u32;
        let mut winding = [0i32; 4];

        let classify_child = |x: f32, y: f32| -> u32 {
            if x <= mid_point.x {
                if y <= mid_point.y {
                    TOP_LEFT
                } else {
                    BOTTOM_LEFT
                }
            } else if y <= mid_point.y {
                TOP_RIGHT
            } else {
                BOTTOM_RIGHT
            }
        };
        let contains_in_parent = |x: f32, y: f32| -> bool {
            x >= bound.left() && x <= bound.right() && y >= bound.top() && y <= bound.bottom()
        };
        // Endpoints inside the parent quad mark their child cell as occupied.
        if contains_in_parent(seg.x0, seg.y0) {
            split_info |= fill(classify_child(seg.x0, seg.y0));
        }
        if contains_in_parent(seg.x1, seg.y1) {
            split_info |= fill(classify_child(seg.x1, seg.y1));
        }

        if einfo.cross0 {
            split_info |= fill(BOTTOM_LEFT);
        }

        if einfo.cross1 {
            split_info |= fill(BOTTOM_RIGHT);
            winding[BOTTOM_LEFT as usize] += going_up;
        }

        if einfo.cross2 {
            split_info |= fill(BOTTOM_LEFT);
        }

        if einfo.cross3 {
            split_info |= fill(BOTTOM_LEFT) | fill(BOTTOM_RIGHT);
            if !einfo.cross16 {
                if !einfo.cross17 {
                    if going_right > 0 {
                        split_info |= up(BOTTOM_LEFT);
                    } else {
                        split_info |= down(BOTTOM_LEFT);
                    }
                } else {
                    winding[BOTTOM_LEFT as usize] += going_right;
                }
            }
        }

        if einfo.cross4 {
            split_info |= fill(BOTTOM_RIGHT);
            if !einfo.cross13 {
                if !einfo.cross12 {
                    if going_right > 0 {
                        split_info |= up(BOTTOM_RIGHT);
                    } else {
                        split_info |= down(BOTTOM_RIGHT);
                    }
                } else {
                    winding[BOTTOM_RIGHT as usize] += going_right;
                }
            }
        }

        if einfo.cross5 {
            split_info |= fill(BOTTOM_LEFT) | fill(TOP_LEFT);
        }

        if einfo.cross6 {
            split_info |= fill(BOTTOM_RIGHT) | fill(TOP_RIGHT);
            winding[TOP_LEFT as usize] += going_up;
        }

        if einfo.cross7 {
            split_info |= fill(TOP_LEFT);
        }

        if einfo.cross8 {
            split_info |= fill(TOP_LEFT) | fill(TOP_RIGHT);
            if !einfo.cross15 {
                if !einfo.cross16 {
                    if going_right > 0 {
                        split_info |= up(TOP_LEFT);
                    } else {
                        split_info |= down(TOP_LEFT);
                    }
                } else {
                    winding[TOP_LEFT as usize] += going_right;
                }
            }
        }

        if einfo.cross9 {
            split_info |= fill(TOP_RIGHT);
            if !einfo.cross14 {
                if !einfo.cross13 {
                    if going_right > 0 {
                        split_info |= up(TOP_RIGHT);
                    } else {
                        split_info |= down(TOP_RIGHT);
                    }
                } else {
                    winding[TOP_RIGHT as usize] += going_right;
                }
            }
        }

        if einfo.cross10 {
            split_info |= fill(TOP_LEFT);
        }

        if einfo.cross11 {
            split_info |= fill(TOP_RIGHT);
        }

        if einfo.cross12 {
            winding[BOTTOM_RIGHT as usize] += going_up;
            winding[BOTTOM_LEFT as usize] += going_up;
        }

        if einfo.cross13 {
            winding[TOP_RIGHT as usize] += going_up;
            winding[TOP_LEFT as usize] += going_up;
        }

        if shortcut != 0 {
            let [x, y] = seg.get_shortcut_base();
            let is_down_shortcut = shortcut == -1;

            if !(y <= bound.top() || x < bound.left()) && x >= bound.right() && y >= mid_point.y {
                winding[TOP_LEFT as usize] += if is_down_shortcut { -1 } else { 1 };
                winding[TOP_RIGHT as usize] += if is_down_shortcut { -1 } else { 1 };

                if y >= bound.bottom() {
                    winding[BOTTOM_LEFT as usize] += if is_down_shortcut { -1 } else { 1 };
                    winding[BOTTOM_RIGHT as usize] += if is_down_shortcut { -1 } else { 1 };
                }
            }
        }

        Self {
            winding,
            split_info,
            _pad: [0; 3],
        }
    }
}

/// Per-entry record stored in a quad cell.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CellEntry {
    pub entry_type: u32,
    pub data: i32,    // WINDING_INCREMENT: increment value; ABSTRACT: shortcut flag
    pub seg_idx: u32, // Index into abs_segments; only valid for ABSTRACT entries
    pub path_idx: u32,
    pub cell_pos: u32,
    pub cell_id: u32,
    pub _pad: [u32; 2],
}

impl Default for CellEntry {
    fn default() -> Self {
        CellEntry {
            entry_type: EMPTY,
            seg_idx: NONE_U32,
            path_idx: u32::MAX,
            data: 0,
            cell_pos: 0,
            cell_id: u32::MAX,
            _pad: [0; 2],
        }
    }
}

/// Intermediate per-entry state used during parallel subdivision (section 4.2).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct SplitEntry {
    split_data: SplitData,
    pub offsets: [u32; 4],
    pub unique_id: u32,
    pub seg_idx: u32,
    pub path_idx: u32,
    /// cell_id of the parent; carries the parent's identity into child entries.
    pub parent_cell_id: u32,
}

/// Build the initial flat list of ABSTRACT entries for the root cell (one per segment).
pub fn init_root_cell_entries(abs_segments: &[AbstractLineSegment]) -> Vec<CellEntry> {
    let mut entries: Vec<_> = vec![];
    for i in 0..abs_segments.len() {
        let curr = &abs_segments[i];
        entries.push(CellEntry {
            entry_type: ABSTRACT,
            seg_idx: i as u32,
            path_idx: curr.path_idx,
            data: 0,
            cell_pos: 0,
            cell_id: 0,
            _pad: [0; 2],
        });
    }
    entries
}

/// Kernel 1 of 4.2 Parallel subdivision
/// Assuming parent_entries already ordered SEGMENTs - WINDING for each cell.
pub fn build_split_entries(
    parent_bound: &Rect,
    mid_point: &Point,
    cell_entries: &mut [CellEntry],
    abs_segments: &[AbstractLineSegment],
) -> Vec<SplitEntry> {
    let mut split_entries: Vec<SplitEntry> = vec![];
    let unique_id = NEXT_CELL_UNIQUE_ID.fetch_add(1, Ordering::Relaxed);

    for entry in &mut *cell_entries {
        let is_abstract_entry = (entry.entry_type & ABSTRACT) != 0;
        let is_winding_inc_entry = (entry.entry_type & WINDING_INCREMENT) != 0;

        if is_abstract_entry {
            let seg_idx = entry.seg_idx;
            let seg = &abs_segments[seg_idx as usize];
            let edge_info = EdgeIntersectionInfo::new(&seg, &parent_bound, &mid_point);
            let split_data =
                SplitData::new(&seg, entry.data, &edge_info, &parent_bound, &mid_point);
            split_entries.push(SplitEntry {
                split_data,
                offsets: [0u32; 4],
                unique_id,
                seg_idx,
                path_idx: seg.path_idx,
                parent_cell_id: entry.cell_id,
            });
        }

        if is_winding_inc_entry {
            let parent_winding = entry.data;
            split_entries.push(SplitEntry {
                split_data: SplitData {
                    winding: [parent_winding; 4],
                    split_info: 0,
                    _pad: [0; 3],
                },
                offsets: [0u32; 4],
                unique_id,
                seg_idx: NONE_U32,
                path_idx: entry.path_idx,
                parent_cell_id: entry.cell_id,
            });
        }
    }
    split_entries
}

/// Kernel 2 of 4.2 Parallel subdivision
/// - Mutate split_entries.
/// - Assuming the Vec<SegmentEntry> and Vec<AbstractLineSegment> is ordered by path already.
/// - Executes inclusive scan to generate the last winding per path, per cell.
pub fn consolidate_winding_inc(split_entries: &mut Vec<SplitEntry>) {
    assert!(split_entries.len() > 0);

    for i in 1..split_entries.len() {
        let prev = split_entries[i - 1];
        let curr = &mut split_entries[i];
        if curr.path_idx == prev.path_idx {
            for cell in 0..4 {
                curr.split_data.winding[cell] += prev.split_data.winding[cell];
            }
        }
    }
}

/// Kernel 3 of 4.2 Parallel subdivision.
/// - Mutate entries.
/// - Assumes entries are ordered by path.
/// - Computes global exclusive offsets in (child -> path -> entry) order.
/// - Outer loop is child cell so that same-cell entries are contiguous in the output,
///   which is required by group_by_cell_pos in the quad-tree builder.
pub fn update_to_global_offset(entries: &mut [SplitEntry]) -> u32 {
    assert!(!entries.is_empty());

    let mut sum: u32 = 0;

    for &cell in &[TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT] {
        let mut start = 0usize;
        while start < entries.len() {
            // Find the start and tail entries in the same path
            let path = entries[start].path_idx;
            let mut end = start + 1;
            while end < entries.len() && entries[end].path_idx == path {
                end += 1;
            }
            let tail = end - 1;

            for i in start..end {
                let split_info = entries[i].split_data.split_info;
                let seg_out = has_fill(split_info, cell) as u32;

                let is_tail = i == tail;
                let winc_out =
                    (is_tail && entries[i].split_data.winding[cell as usize] != 0) as u32;

                entries[i].offsets[cell as usize] = sum;
                sum += seg_out + winc_out;
            }

            start = end;
        }
    }
    sum
}

/// Kernel 4 — scatter split entries into child `CellEntry` records.
pub fn split_to_cell_entry(split_entries: &mut [SplitEntry], out_vec_size: u32) -> Vec<CellEntry> {
    assert!(split_entries.last().is_some());
    let mut cell_entries: Vec<CellEntry> = vec![CellEntry::default(); out_vec_size as usize];

    for &cell in &[TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT] {
        let ci = cell as usize;
        let mut start = 0;
        while start < split_entries.len() {
            let path = split_entries[start].path_idx;
            let mut end = start + 1;
            while end < split_entries.len() && split_entries[end].path_idx == path {
                end += 1;
            }
            let tail = end - 1;

            for i in start..end {
                let curr = &split_entries[i];
                let next = if i + 1 < end {
                    Some(&split_entries[i + 1])
                } else {
                    None
                };
                if next.is_some_and(|next| next.offsets[ci] == curr.offsets[ci]) {
                    continue;
                }

                let has_segment = has_fill(curr.split_data.split_info, cell);
                let has_winding = (i == tail) && curr.split_data.winding[ci] != 0;
                let shortcut = if has_up(curr.split_data.split_info, cell) {
                    1
                } else if has_down(curr.split_data.split_info, cell) {
                    -1
                } else {
                    0
                };

                let base = curr.offsets[ci] as usize;
                let mut cursor = base;
                if has_segment {
                    cell_entries[cursor] = CellEntry {
                        entry_type: ABSTRACT,
                        data: shortcut,
                        seg_idx: curr.seg_idx,
                        path_idx: curr.path_idx,
                        cell_pos: cell,
                        cell_id: curr.parent_cell_id * 4 + cell,
                        _pad: [0; 2],
                    };
                    cursor += 1;
                }
                if has_winding {
                    cell_entries[cursor] = CellEntry {
                        entry_type: WINDING_INCREMENT,
                        data: curr.split_data.winding[ci],
                        seg_idx: NONE_U32,
                        path_idx: curr.path_idx,
                        cell_pos: cell,
                        cell_id: curr.parent_cell_id * 4 + cell,
                        _pad: [0; 2],
                    };
                }
            }

            start = end;
        }
    }
    cell_entries
}

/// Execute Kernel 1 ~ 4 of 4.2 Parallel Subdivision on CPU.
pub fn subdivide_cell_entry(
    cell_entries: &mut [CellEntry],
    parent_bound: &Rect,
    parent_mid_point: &Point,
    abs_segments: &[AbstractLineSegment],
) -> anyhow::Result<Vec<CellEntry>> {
    let mut split_entries =
        build_split_entries(parent_bound, parent_mid_point, cell_entries, abs_segments);
    consolidate_winding_inc(&mut split_entries);
    let out_vec_size = update_to_global_offset(&mut split_entries);
    let next_cell_entries = split_to_cell_entry(&mut split_entries, out_vec_size);
    Ok(next_cell_entries)
}

fn print_entries<T: Debug>(entries: &[T], mut cell_pos: impl FnMut(&T) -> u32) {
    let mut last = None::<u8>;
    for e in entries {
        let pos = cell_pos(e);
        if last.is_some_and(|p| p as u32 != pos) {
            println!();
        }
        println!("{:?}", e);
        last = Some(pos as u8);
    }
}

pub fn print_split_entries(entries: &[SplitEntry]) {
    for e in entries {
        print!("seg_idx: {:?}, ", e.seg_idx);
        print!("path_id: {:?}, ", e.path_idx);
        print!("offsets: {:?}, ", e.offsets);
        print!("unique_id: {:?}", e.unique_id);
        println!();
        print_split_data(&e.split_data);
        println!();
        println!();
    }
}

fn print_split_data(split_data: &SplitData) {
    for cell in 0..4 {
        let has_segment = has_fill(split_data.split_info, cell);
        let shortcut = if has_up(split_data.split_info, cell) {
            1
        } else if has_down(split_data.split_info, cell) {
            -1
        } else {
            0
        };
        print!(
            "{} [",
            match cell {
                0 => {
                    "TL"
                }
                1 => {
                    "TR"
                }
                2 => {
                    "BL"
                }
                3 => {
                    "BR"
                }
                _ => {
                    "Invalid"
                }
            }
        );
        print!("seg: {}, ", has_segment);
        print!("winc: {}, ", split_data.winding[cell as usize]);
        print!("short: {}", shortcut);
        print!("] ");
    }
}
