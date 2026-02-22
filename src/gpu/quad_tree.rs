use crate::abstract_segment::AbstractLineSegment;
use crate::seg_entry::SegEntry;
use crate::geometry::rect::Rect;
use crate::gpu::subdivide_seg_entry::QuadTreeGpuContext;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CellMetadata {
    bbox_ltrb: [f32; 4],
    mid: [f32; 2],
    entry_start: u32,
    entry_count: u32,
    abstract_count: u32,
    _pad: [u32; 3],
}

impl CellMetadata {
    pub fn new(rect: &Rect, entry_start: u32, entry_count: u32) -> Self {
        Self {
            bbox_ltrb: [rect.left(), rect.top(), rect.right(), rect.bottom()],
            mid: rect.mid_point(),
            entry_start,
            entry_count,
            // Root starts from abstract entries only; child cells are updated on GPU.
            abstract_count: entry_count,
            _pad: [0; 3],
        }
    }

    pub fn bbox_ltrb(&self) -> [f32; 4] {
        self.bbox_ltrb
    }

    pub fn entry_start(&self) -> u32 {
        self.entry_start
    }

    pub fn entry_count(&self) -> u32 {
        self.entry_count
    }

    pub fn bbox_rect(&self) -> Rect {
        Rect::from_ltrb(
            self.bbox_ltrb[0],
            self.bbox_ltrb[1],
            self.bbox_ltrb[2],
            self.bbox_ltrb[3],
        )
        .expect("CellMetadata bbox_ltrb should always be valid")
    }
}
pub fn build_quadtree(
    root_bbox: Rect,
    root_entries: Vec<SegEntry>,
    max_depth: u8,
    min_seg: usize,
    abs_segments: &[AbstractLineSegment],
) -> anyhow::Result<(Vec<CellMetadata>, Vec<SegEntry>)> {
    let gpu_ctx = pollster::block_on(QuadTreeGpuContext::new(
        &root_entries,
        abs_segments,
        &root_bbox,
        max_depth,
        min_seg as u32,
    ))?;

    let mut num_cells = 1u32;
    let mut num_entries = root_entries.len() as u32;

    for depth in 0..max_depth {
        gpu_ctx.process_level(depth, num_cells, num_entries);

        // Read back the actual output entry count; needed because the GPU emits a
        // variable number of entries and the next dispatch must use the correct size.
        let result_info = gpu_ctx.read_result_info()?;
        num_entries = result_info.seg_entries_length;
        num_cells *= 4;
    }

    let mut result_seg_entries = gpu_ctx.read_seg_entry()?;
    // Last depth processed is max_depth - 1; pass it to select the correct ping-pong buffer.
    let last_depth = max_depth - 1;
    let cell_metadata = gpu_ctx.read_cell_metadata(last_depth)?;

    // num_entries was updated to the final level's output count after the last readback.
    result_seg_entries.truncate(num_entries as usize);
    Ok((cell_metadata, result_seg_entries))
}
