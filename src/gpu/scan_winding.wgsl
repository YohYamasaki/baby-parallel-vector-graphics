const WG_SIZE: u32 = 2u;

const NONE_U32: u32 = 0xFFFFFFFF;
const TOP_LEFT: u32 = 0;
const TOP_RIGHT: u32 = 1;
const BOTTOM_LEFT: u32 = 2;
const BOTTOM_RIGHT: u32 = 3;

const ABSTRACT: u32 = 1 << 0;
const WINDING_INCREMENT: u32 = 1 << 3;

struct ParentCellBound {
    bbox_ltrb: vec4<f32>,
    mid_x: f32,
    mid_y: f32,
    _pad: vec2<u32>
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

struct CellEntry {
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
    _pad: u32
}

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

/// Kernel 2 of 4.2 Parallel subdivision
/// - Mutate split_entries.
/// - Assuming the Vec<SegmentEntry> and Vec<AbstractLineSegment> is ordered by path already.
/// - Executes inclusive scan to generate the last winding per path, per cell.
fn inclusive_scan_winding_inc(lid: u32) {
    // Hillis-Steele inclusive scan
    var offset = 1u;
    loop {
        if (offset >= WG_SIZE) { break; }
        var add_val = vec4<i32>(0, 0, 0, 0);
        // TODO: do we have to check if the segments in the same path?
        if (lid >= offset && split_entries[lid - offset].path_idx == split_entries[lid].path_idx) {
            add_val = split_entries[lid - offset].split_data.winding;
        }
        workgroupBarrier();

        split_entries[lid].split_data.winding += add_val;
        workgroupBarrier();

        offset *= 2u;
    }
}

@group(0) @binding(0) var<storage, read_write> cell_entries: array<CellEntry>;
@group(0) @binding(1) var<storage, read_write> global_split_entries: array<SplitEntry>;
@group(0) @binding(2) var<storage, read_write> global_cell_offsets: array<u32>;

var<workgroup> split_entries: array<SplitEntry, 2>;
var<workgroup> cell_offsets: array<u32, 256>; // TODO: Max offsets so far, 64 * 4 cells

@compute
@workgroup_size(WG_SIZE)
fn scan_winding_block(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    let entries_length = arrayLength(&cell_entries);
    let in_range = idx < entries_length;

    if (in_range) {
        split_entries[lid.x] = global_split_entries[idx];
    }

    workgroupBarrier();

    inclusive_scan_winding_inc(lid.x);

    if (in_range) {
        global_split_entries[idx] = split_entries[lid.x];
    }
}


@compute
@workgroup_size(WG_SIZE)
fn winding_block_sums(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    let entries_length = arrayLength(&cell_entries);
    let in_range = idx < entries_length;

    if (in_range) {
        split_entries[lid.x] = global_split_entries[idx];
    }

    workgroupBarrier();

    if (in_range) {
        global_split_entries[idx] = split_entries[lid.x];
    }
}

@compute
@workgroup_size(2)
fn mark_tail_winding(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
){
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    let entries_length = arrayLength(&cell_entries);
    let in_range = idx < entries_length;
    
    if (in_range) {
        split_entries[lid.x] = global_split_entries[idx];
        for (var cell = 0u; cell < 4u; cell++) {
            cell_offsets[cell * entries_length + idx] = global_cell_offsets[cell * entries_length + idx];
        }
    }

    // insert additional offset for winding increment entry
    if (in_range) {
        var is_path_tail = idx == entries_length - 1u;
        if (!is_path_tail) {
            is_path_tail = cell_entries[idx + 1u].path_idx != cell_entries[idx].path_idx;
        }
        if (is_path_tail) {
            for (var cell = 0u; cell < 4u; cell++) {
                let last_winc_in_child = split_entries[lid.x].split_data.winding[cell];
                if (last_winc_in_child != 0) {
                    cell_offsets[cell * entries_length + idx] += 1;
                }
            }
        }
    }
    workgroupBarrier();

    if (in_range) {
        global_split_entries[idx] = split_entries[lid.x];
    }

    if (lid.x == 0u) {
        for (var i = 0u; i < entries_length * 4u; i++) {
            global_cell_offsets[i] = cell_offsets[i];
        }
    }
}