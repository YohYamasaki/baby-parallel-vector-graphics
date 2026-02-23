const NONE_U32: u32 = 0xFFFFFFFF;
const TOP_LEFT: u32 = 0;
const TOP_RIGHT: u32 = 1;
const BOTTOM_LEFT: u32 = 2;
const BOTTOM_RIGHT: u32 = 3;

const ABSTRACT: u32 = 1 << 0;
const WINDING_INCREMENT: u32 = 1 << 3;

struct CellMetadata {
    bbox_ltrb: vec4<f32>,
    mid: vec2<f32>,
    entry_start: u32,
    entry_count: u32,
    abstract_count: u32,
    _pad: array<u32, 3>,
}

struct AbstractLineSegment {
    seg_type: u32,
    path_idx: u32,
    _pad0: vec2<u32>,
    bbox_ltrb: vec4<f32>,
    direction: u32,
    a: f32,
    b: f32,
    c: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

struct SegEntry {
    entry_type: u32,
    data: i32,    // winding -> winding increment, segment -> shortcut
    seg_idx: u32, // For abstract entry
    path_idx: u32,
    cell_pos: u32, // Use BOTTOM_LEFT ~ TOP_RIGHT
    cell_id: u32,  // This will be provided after cell entry subdivision
    _pad: array<u32, 2>
}

struct SplitData {
    winding: vec4<i32>,
    split_info: u32,
    _pad: array<u32, 3>
}

struct SplitEntry {
    split_data: SplitData,
    offsets: vec4<u32>,
    unique_id: u32,
    seg_idx: u32, // For abstract entry
    path_idx: u32,
    parent_cell_id: u32, // Propagated from entry.cell_id to track parent cell across subdivision
}

struct WindingBlockInfo {
    first_path_idx: u32,
    last_path_idx: u32,
    first_cell_id: u32,
    last_cell_id: u32,
    tail_winding: vec4<i32>,
}

struct SplitResultInfo {
    seg_entries_length: u32,
    min_seg: u32,
    _pad: vec2<u32>,
}

struct ScanParams {
    level_len: u32,
    carry_len: u32,
    _pad: vec2<u32>,
}

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}
