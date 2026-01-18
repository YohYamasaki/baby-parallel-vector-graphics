use crate::abstract_segment::AbstractLineSegment;
use crate::cell_entry::{ABSTRACT, CellEntry, init_root_cell_entries, subdivide_cell_entry};
use crate::geometry::rect::Rect;
use std::ops::Range;
use usvg::tiny_skia_path::Point;

pub const TL_IDX: u32 = 0;
pub const TR_IDX: u32 = 1;
pub const BL_IDX: u32 = 2;
pub const BR_IDX: u32 = 3;

#[derive(Debug, Copy, Clone)]
pub struct CellSegmentRef {
    pub seg_index: usize,
    pub shortcut: i8,
}
#[derive(Debug, Clone)]
pub struct QuadCell {
    pub id: crate::cell_entry::CellId,
    pub depth: u8,
    pub bbox: Rect,
    pub children: Option<[crate::cell_entry::CellId; 4]>,
    // This data will be filled once this cell is confirmed as a leaf
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

/// Build QuadTree from root CellEntry by level order.
fn build_quadtree(
    root_bbox: Rect,
    mut root_entries: Vec<CellEntry>,
    max_depth: u8,
    min_seg: usize,
    abs_segments: &[AbstractLineSegment],
) -> anyhow::Result<(Vec<QuadCell>, Vec<CellEntry>)> {
    // node arena; this will hold all cells' metadata
    let mut nodes: Vec<QuadCell> = Vec::new();
    let root_id: crate::cell_entry::CellId = 0;
    // add root node first
    nodes.push(QuadCell {
        id: root_id,
        depth: 0,
        bbox: root_bbox,
        children: None,
        leaf_entry_range: None,
    });

    // Add cell_id to root entries
    for e in &mut root_entries {
        e.cell_id = root_id;
        e.cell_pos = TL_IDX; // Dummy
    }

    // frontier (cells to be subdivided)
    let mut cells_cur: Vec<crate::cell_entry::CellId> = vec![root_id];
    let mut entries_cur: Vec<CellEntry> = root_entries;
    let mut ranges_cur: Vec<Range<usize>> = vec![0..entries_cur.len()];

    // leaf stores cells that are already done subdivision
    let mut leaf_entries: Vec<CellEntry> = Vec::new();

    for depth in 0..max_depth {
        if cells_cur.is_empty() {
            break;
        }

        let mut cells_next: Vec<crate::cell_entry::CellId> = Vec::new();
        let mut entries_next: Vec<CellEntry> = Vec::new();
        let mut ranges_next: Vec<Range<usize>> = Vec::new();

        for (i, &parent_id) in cells_cur.iter().enumerate() {
            let parent_range = ranges_cur[i].clone(); // for one cell
            let parent_entries = &mut entries_cur[parent_range.clone()];
            let parent_seg_count = parent_entries
                .iter()
                .filter(|e| (e.entry_type & ABSTRACT) != 0)
                .count();
            let can_split = parent_seg_count > min_seg;
            let parent_bbox = nodes[parent_id as usize].bbox;

            let child_bounds = if can_split {
                let mid = Point {
                    x: (parent_bbox.right() + parent_bbox.left()) * 0.5,
                    y: (parent_bbox.bottom() + parent_bbox.top()) * 0.5,
                };
                let Some(cb) = get_child_bounds(parent_bbox, mid) else {
                    // bounding box is not valid, consider as a leaf
                    let start = leaf_entries.len();
                    leaf_entries.extend_from_slice(parent_entries);
                    nodes[parent_id as usize].leaf_entry_range = Some(start..leaf_entries.len());
                    continue;
                };
                (mid, cb)
            } else {
                // Cannot subdivide any more, save this cell as a leaf
                let start = leaf_entries.len();
                leaf_entries.extend_from_slice(parent_entries);
                nodes[parent_id as usize].leaf_entry_range = Some(start..leaf_entries.len());
                continue;
            };

            let (mid_point, child_bounds) = child_bounds;

            // Create 4 child cells in arena
            let child_ids: [crate::cell_entry::CellId; 4] = std::array::from_fn(|pos| {
                let id = nodes.len() as crate::cell_entry::CellId;
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

            // Subdivide parent's cell entries into child cell entries, then create contiguous entries/ranges
            let mut child_entries_flat =
                subdivide_cell_entry(parent_entries, &parent_bbox, &mid_point, abs_segments)?;

            // 子ごとに仕分けして、next の contiguous な entries/ranges を作る
            // TODO: is this splitting actually required??
            let mut entries_per_child: [Vec<CellEntry>; 4] = std::array::from_fn(|_| Vec::new());
            for mut e in child_entries_flat.drain(..) {
                // TODO: drain必要？
                let pos = e.cell_pos as usize; // 0..3
                e.cell_id = child_ids[pos]; // set actual child cell id
                entries_per_child[pos].push(e);
            }

            for pos in 0..4 {
                if entries_per_child[pos].is_empty() {
                    continue;
                }

                let start = entries_next.len();
                entries_next.extend_from_slice(&entries_per_child[pos]);
                let end = entries_next.len();

                cells_next.push(child_ids[pos]);
                ranges_next.push(start..end);
            }
        }

        cells_cur = cells_next;
        entries_cur = entries_next;
        ranges_cur = ranges_next;
    }

    // Add all the frontiers to the leaf vec as the maximum depth is reached
    for (i, &cell_id) in cells_cur.iter().enumerate() {
        let range = ranges_cur[i].clone();
        let start = leaf_entries.len();
        leaf_entries.extend_from_slice(&entries_cur[range]);
        nodes[cell_id as usize].leaf_entry_range = Some(start..leaf_entries.len());
    }

    Ok((nodes, leaf_entries))
}

fn get_child_bounds(parent_bbox: Rect, mid: Point) -> Option<[Rect; 4]> {
    let tl = Rect::from_ltrb(parent_bbox.left(), parent_bbox.top(), mid.x, mid.y)?;
    let tr = Rect::from_ltrb(mid.x, parent_bbox.top(), parent_bbox.right(), mid.y)?;
    let bl = Rect::from_ltrb(parent_bbox.left(), mid.y, mid.x, parent_bbox.bottom())?;
    let br = Rect::from_ltrb(mid.x, mid.y, parent_bbox.right(), parent_bbox.bottom())?;
    Some([tl, tr, bl, br])
}
