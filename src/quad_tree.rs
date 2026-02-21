use crate::abstract_segment::AbstractLineSegment;
use crate::cell_entry::{
    init_root_cell_entries, subdivide_cell_entry, CellEntry, CellId, ABSTRACT,
};
use crate::geometry::rect::Rect;
use std::ops::Range;
use usvg::tiny_skia_path::Point;

#[derive(Debug, Copy, Clone)]
pub struct CellSegmentRef {
    pub seg_index: usize,
    pub shortcut: i8,
}
#[derive(Debug, Clone)]
pub struct QuadCell {
    pub id: CellId,
    pub depth: u8,
    pub bbox: Rect,
    pub children: Option<[CellId; 4]>,
    /// Set once this cell is finalised as a leaf.
    pub leaf_entry_range: Option<Range<usize>>,
}
#[derive(Debug)]
pub struct QuadTree {
    pub nodes: Vec<QuadCell>,
    pub entries: Vec<CellEntry>,
}

impl QuadTree {
    pub fn new(
        abs_segments: &[AbstractLineSegment],
        root_bbox: Rect,
        max_depth: u8,
        min_seg: usize,
    ) -> anyhow::Result<Self> {
        let root_entries = init_root_cell_entries(&abs_segments);
        let (nodes, entries) =
            build_quadtree(root_bbox, root_entries, max_depth, min_seg, abs_segments)?;
        Ok(Self { nodes, entries })
    }
}

/// Build a quad tree by level-order subdivision.
///
/// Each level processes the current frontier, subdividing cells that have more
/// than `min_seg` ABSTRACT entries and marking the rest as leaves.
fn build_quadtree(
    root_bbox: Rect,
    root_entries: Vec<CellEntry>,
    max_depth: u8,
    min_seg: usize,
    abs_segments: &[AbstractLineSegment],
) -> anyhow::Result<(Vec<QuadCell>, Vec<CellEntry>)> {
    let mut nodes: Vec<QuadCell> = Vec::new();
    let mut leaf_entries: Vec<CellEntry> = Vec::new();

    // Root node
    let root_id: CellId = 0;
    nodes.push(QuadCell {
        id: root_id,
        depth: 0,
        bbox: root_bbox,
        children: None,
        leaf_entry_range: None,
    });

    // Frontier: list of (node_id, owned entries) pairs to process at each level.
    // When moving to GPU, replace with a flat buffer + metadata array.
    let mut frontier: Vec<(CellId, Vec<CellEntry>)> = vec![(root_id, root_entries)];

    for depth in 0..max_depth {
        if frontier.is_empty() {
            break;
        }

        let mut next_frontier: Vec<(CellId, Vec<CellEntry>)> = Vec::new();

        for (parent_id, mut parent_entries) in frontier {
            let abstract_count = parent_entries
                .iter()
                .filter(|e| (e.entry_type & ABSTRACT) != 0)
                .count();
            let parent_bbox = nodes[parent_id as usize].bbox;

            // Decide whether this cell needs further subdivision
            if abstract_count <= min_seg {
                save_as_leaf(&mut nodes, &mut leaf_entries, parent_id, parent_entries);
                continue;
            }

            let [mid_x, mid_y] = parent_bbox.mid_point();
            let mid = Point { x: mid_x, y: mid_y };
            let Some(child_bounds) = get_child_bounds(parent_bbox, mid) else {
                save_as_leaf(&mut nodes, &mut leaf_entries, parent_id, parent_entries);
                continue;
            };

            // --- Create child nodes ---
            let child_ids: [CellId; 4] = std::array::from_fn(|pos| {
                let id = nodes.len() as CellId;
                nodes.push(QuadCell {
                    id,
                    depth: depth + 1,
                    bbox: child_bounds[pos],
                    children: None,
                    leaf_entry_range: None,
                });
                id
            });
            nodes[parent_id as usize].children = Some(child_ids);

            let child_entries =
                subdivide_cell_entry(&mut parent_entries, &parent_bbox, &mid, abs_segments)?;

            // subdivide output is already in (TL, TR, BL, BR) order.
            for (child_id, entries) in group_by_cell_pos(child_entries, &child_ids) {
                if !entries.is_empty() {
                    next_frontier.push((child_id, entries));
                }
            }
        }

        frontier = next_frontier;
    }

    // Remaining frontier cells reached max depth; finalize them as leaves.
    for (cell_id, entries) in frontier {
        save_as_leaf(&mut nodes, &mut leaf_entries, cell_id, entries);
    }

    Ok((nodes, leaf_entries))
}

/// Mark a cell as a leaf and append its entries to the global leaf entry list.
fn save_as_leaf(
    nodes: &mut Vec<QuadCell>,
    leaf_entries: &mut Vec<CellEntry>,
    cell_id: CellId,
    entries: Vec<CellEntry>,
) {
    let start = leaf_entries.len();
    leaf_entries.extend(entries);
    nodes[cell_id as usize].leaf_entry_range = Some(start..leaf_entries.len());
}

/// Partition a flat entry list into per-child groups based on `cell_pos`.
///
/// Assumes entries are contiguous per cell in order TL(0), TR(1), BL(2), BR(3).
fn group_by_cell_pos(
    entries: Vec<CellEntry>,
    child_ids: &[CellId; 4],
) -> Vec<(CellId, Vec<CellEntry>)> {
    let mut result: Vec<(CellId, Vec<CellEntry>)> = Vec::new();
    let mut current_pos: Option<usize> = None;
    let mut current_group: Vec<CellEntry> = Vec::new();

    for mut entry in entries {
        let pos = entry.cell_pos as usize;
        if current_pos != Some(pos) {
            if let Some(prev_pos) = current_pos {
                result.push((child_ids[prev_pos], std::mem::take(&mut current_group)));
            }
            current_pos = Some(pos);
        }
        entry.cell_id = child_ids[pos];
        current_group.push(entry);
    }
    if let Some(pos) = current_pos {
        result.push((child_ids[pos], current_group));
    }
    result
}

fn get_child_bounds(parent_bbox: Rect, mid: Point) -> Option<[Rect; 4]> {
    let tl = Rect::from_ltrb(parent_bbox.left(), parent_bbox.top(), mid.x, mid.y)?;
    let tr = Rect::from_ltrb(mid.x, parent_bbox.top(), parent_bbox.right(), mid.y)?;
    let bl = Rect::from_ltrb(parent_bbox.left(), mid.y, mid.x, parent_bbox.bottom())?;
    let br = Rect::from_ltrb(mid.x, mid.y, parent_bbox.right(), parent_bbox.bottom())?;
    Some([tl, tr, bl, br])
}
