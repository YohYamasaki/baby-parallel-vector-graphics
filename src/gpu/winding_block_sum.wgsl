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
    seg_idx: u32,
    path_idx: u32,
    parent_cell_id: u32,
}

struct WindingBlockInfo {
    first_path_idx: u32,
    last_path_idx: u32,
    first_cell_id: u32,
    last_cell_id: u32,

    tail_winding: vec4<i32>,
}

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

fn neutral_winc() -> WindingBlockInfo {
    var z = WindingBlockInfo();
    z.first_path_idx = NONE_U32;
    z.last_path_idx = NONE_U32;
    z.first_cell_id = NONE_U32;
    z.last_cell_id = NONE_U32;
    z.tail_winding = vec4<i32>(0, 0, 0, 0);
    return z;
}

fn merge(a: WindingBlockInfo, b: WindingBlockInfo) -> WindingBlockInfo {
    var out = b;

    let same_boundary = (a.last_path_idx == b.first_path_idx) && (a.last_cell_id == b.first_cell_id);
    let b_is_single_group =
        (b.first_path_idx == b.last_path_idx) &&
        (b.first_cell_id == b.last_cell_id);
    if (same_boundary && b_is_single_group) {
        out.tail_winding = a.tail_winding + b.tail_winding;
    }
    return out;
}

// Kernel 2 — Hillis-Steele inclusive scan to accumulate the final winding per (path, cell).
fn inclusive_scan_winding_inc(lid: u32) {
    // Hillis-Steele inclusive scan
    var offset = 1u;
    loop {
        if (offset >= WG_SIZE) { break; }
        var new_winc_info = WindingBlockInfo();
        if (lid >= offset) {
            new_winc_info = merge(wincs[lid - offset], wincs[lid]);
        } else {
            new_winc_info = wincs[lid];
        }
        workgroupBarrier();

        wincs[lid] = new_winc_info;
        workgroupBarrier();

        offset *= 2u;
    }
}

struct SplitResultInfo {
    seg_entries_length: u32,
}

struct ScanParams {
    level_len: u32,
    carry_len: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> seg_entries: array<SegEntry>;
@group(0) @binding(1) var<storage, read_write> global_split_entries: array<SplitEntry>;
@group(0) @binding(2) var<storage, read_write> global_cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> winding_infos_1: array<WindingBlockInfo>;
@group(0) @binding(4) var<storage, read_write> winding_infos_2: array<WindingBlockInfo>; // per-block summaries
// result_info[0].seg_entries_length holds the actual number of entries for the current depth.
@group(0) @binding(5) var<storage, read_write> result_info: array<SplitResultInfo>;
@group(0) @binding(6) var<storage, read_write> scan_params: array<ScanParams>;

var<workgroup> wincs: array<WindingBlockInfo, 2>;

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
    let entries_length = scan_params[0].level_len;
    let carry_len = scan_params[0].carry_len;
    let block_start = wg_linear * WG_SIZE;

    var block_len = 0u;
    if (block_start < entries_length) {
        let remaining = entries_length - block_start;
        block_len = min(WG_SIZE, remaining);
    }

    if (block_len == 0u) {
        return;
    }

    let in_range = idx < entries_length;

    if (in_range) {
        wincs[lid.x] = winding_infos_1[idx];
    } else {
        wincs[lid.x] = neutral_winc();
    }
    workgroupBarrier();

    inclusive_scan_winding_inc(lid.x);
    workgroupBarrier();

    if (lid.x == 0u && wg_linear < carry_len) {
        let last_valid_idx = block_len - 1u;
        var block_sum = WindingBlockInfo();
        block_sum.first_path_idx = wincs[0].first_path_idx;
        block_sum.last_path_idx = wincs[last_valid_idx].last_path_idx;
        block_sum.first_cell_id = wincs[0].first_cell_id;
        block_sum.last_cell_id = wincs[last_valid_idx].last_cell_id;
        block_sum.tail_winding = wincs[last_valid_idx].tail_winding;
        winding_infos_2[wg_linear] = block_sum;
    }

    if (in_range) {
        winding_infos_1[idx] = wincs[lid.x];
    }
}

@compute
@workgroup_size(WG_SIZE)
fn add_winding_carry(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    let entries_length = scan_params[0].level_len;
    let in_range = idx < entries_length;

    if (!in_range || wg_linear == 0u) {
        return;
    }

    let carry_len = scan_params[0].carry_len;
    let carry_idx = wg_linear - 1u;
    if (carry_idx >= carry_len) {
        return;
    }

    let carry = winding_infos_2[carry_idx];
    var curr = winding_infos_1[idx];
    let curr_is_single_group =
        (curr.first_path_idx == curr.last_path_idx) &&
        (curr.first_cell_id == curr.last_cell_id);
    if (curr_is_single_group &&
        carry.last_path_idx == curr.first_path_idx &&
        carry.last_cell_id == curr.first_cell_id) {
        curr.tail_winding += carry.tail_winding;
    }
    winding_infos_1[idx] = curr;
}


@compute
@workgroup_size(WG_SIZE)
fn mark_tail_winding(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    let entries_length = result_info[0].seg_entries_length;
    let in_range = idx < entries_length;

    if (!in_range) {
        return;
    }

    var split_entry = global_split_entries[idx];
    split_entry.split_data.winding = winding_infos_1[idx].tail_winding;

    var is_path_tail = idx == entries_length - 1u;
    if (!is_path_tail) {
        is_path_tail =
            seg_entries[idx + 1u].path_idx != seg_entries[idx].path_idx ||
            seg_entries[idx + 1u].cell_id != seg_entries[idx].cell_id;
    }
    if (is_path_tail) {
        for (var cell = 0u; cell < 4u; cell++) {
            let last_winc_in_child = split_entry.split_data.winding[cell];
            if (last_winc_in_child != 0) {
                global_cell_offsets[cell * entries_length + idx] += 1u;
            }
        }
    }

    global_split_entries[idx] = split_entry;
}
