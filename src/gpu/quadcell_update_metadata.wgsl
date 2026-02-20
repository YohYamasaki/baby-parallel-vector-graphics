const WG_SIZE: u32 = 1u;

struct CellMetadata {
    bbox_ltrb: vec4<f32>,
    mid: vec2<f32>,
    entry_start: u32,
    entry_count: u32,
}

struct CellEntry {
    entry_type: u32,
    data: i32,    // winding -> winding increment, segment -> shortcut
    seg_idx: u32, // For abstract entry
    path_idx: u32,
    cell_pos: u32, // Use BOTTOM_LEFT ~ TOP_RIGHT
    cell_id: u32,  // This will be provided after cell entry subdivision
    _pad: array<u32, 2>
}

struct SplitResultInfo {
    cell_entries_length: u32,
}

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

fn get_child_bounds(parent_bbox: vec4<f32>, mid_x: f32, mid_y: f32) -> array<vec4<f32>, 4> {
    let p_left = parent_bbox[0];
    let p_top = parent_bbox[1];
    let p_right = parent_bbox[2];
    let p_bottom = parent_bbox[3];
    let tl = vec4(p_left, p_top, mid_x, mid_y);
    let tr = vec4(mid_x, p_top, p_right, mid_y);
    let bl = vec4(p_left, mid_y, mid_x, p_bottom);
    let br = vec4(mid_x, mid_y, p_right, p_bottom);
    return array(tl, tr, bl, br);
}

@group(0) @binding(0) var<storage, read> cell_entries: array<CellEntry>;
@group(0) @binding(1) var<storage, read_write> cell_metadata: array<CellMetadata>;
@group(0) @binding(2) var<storage, read_write> result_info: array<SplitResultInfo>;


@compute
@workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let entry_idx = linearize_workgroup_id(wid, num_wg);
    let result_len = result_info[0].cell_entries_length;

    if (entry_idx >= result_len) {
        return;
    }

    let entry = cell_entries[entry_idx];
    let cell_id = entry.cell_id;

    if (entry_idx == 0 || cell_entries[entry_idx - 1].cell_id != cell_id) {
        cell_metadata[cell_id].entry_start = entry_idx;

        // Find the last entry position by loop, this might be faster to do by another dispatch
        var end = entry_idx + 1u;
        while (end < result_len && cell_entries[end].cell_id == cell_id) {
            end++;
        }
        cell_metadata[cell_id].entry_count = end - entry_idx;
    }
}
