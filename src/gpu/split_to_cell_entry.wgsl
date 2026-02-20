const WG_SIZE: u32 = 2u;

const NONE_U32: u32 = 0xFFFFFFFF;
const TOP_LEFT: u32 = 0;
const TOP_RIGHT: u32 = 1;
const BOTTOM_LEFT: u32 = 2;
const BOTTOM_RIGHT: u32 = 3;

const ABSTRACT: u32 = 1 << 0;
const WINDING_INCREMENT: u32 = 1 << 3;

struct SplitResultInfo {
    cell_entries_length: u32,
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
struct ParentCellBound {
    bbox_ltrb: vec4<f32>,
    mid_y: f32,
    mid_x: f32,
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

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

fn flag(x: u32, offset: u32) -> u32 {
    return (1u << x) << offset;
}

fn fill(x: u32) -> u32 {
    return flag(x, 0);
}

fn has_fill(split_info: u32, cell: u32) -> u32 {
    return select(0u, 1u, (split_info & fill(cell)) != 0);
}

fn up(x: u32) -> u32 {
    return flag(x, 4);
}

fn has_up(split_info: u32, cell: u32) -> bool {
    return (split_info & up(cell)) != 0;
}

fn down(x: u32) -> u32 {
    return flag(x, 8);
}

fn has_down(split_info: u32, cell: u32) -> bool {
    return (split_info & down(cell)) != 0;
}

@group(0) @binding(0) var<storage, read_write> cell_entries: array<CellEntry>;
@group(0) @binding(1) var<storage, read_write> split_entries: array<SplitEntry>;
@group(0) @binding(2) var<storage, read_write> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> result_info: array<SplitResultInfo>;

@compute
@workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let offset_idx = wg_linear * WG_SIZE + lid.x; // global idx for cell_offsets

    let num_entries = result_info[0].cell_entries_length;
    if (num_entries == 0u) {
        return;
    }
    let offsets_length = num_entries * 4u;
    let in_range = offset_idx < offsets_length;
    let split_idx  = offset_idx % num_entries;
    let cell_pos = offset_idx / num_entries;

    if (in_range) {
        let split_entry = split_entries[split_idx];
        let split_info = split_entry.split_data.split_info;
        let curr_offset = cell_offsets[offset_idx];

        var is_path_tail = split_idx == num_entries - 1u;
        if (!is_path_tail) {
            let next_entry = split_entries[split_idx + 1u];
            is_path_tail =
                next_entry.path_idx != split_entry.path_idx ||
                next_entry.parent_cell_id != split_entry.parent_cell_id;
        }

        let emit_seg = has_fill(split_info, cell_pos) == 1u;
        let winc = split_entry.split_data.winding[cell_pos];
        let emit_winc = is_path_tail && (winc != 0);

        let lane_count = select(0u, 1u, emit_seg) + select(0u, 1u, emit_winc);
        if (lane_count > 0u) {
            // cell_offsets now stores an inclusive scan result.
            var out_idx = curr_offset - lane_count;
            var cell_entry = CellEntry();

            // Build CellEntry
            if (emit_seg) {
                // Abstract segment entry
                var shortcut = 0;
                if (has_up(split_info, cell_pos)) {
                    shortcut = 1;
                } else if (has_down(split_info, cell_pos)) {
                    shortcut = -1;
                }
                cell_entry.entry_type = ABSTRACT;
                cell_entry.data = shortcut;
                cell_entry.seg_idx = split_entry.seg_idx;
                cell_entry.path_idx = split_entry.path_idx;
                cell_entry.cell_pos = cell_pos;
                cell_entry.cell_id = split_entry.parent_cell_id * 4u + cell_pos;
                cell_entries[out_idx] = cell_entry;
                out_idx++;
            }

            // Winding increment entry
            if (emit_winc) {
                cell_entry.entry_type = WINDING_INCREMENT;
                cell_entry.data = winc;
                cell_entry.seg_idx = NONE_U32;
                cell_entry.path_idx = split_entry.path_idx;
                cell_entry.cell_pos = cell_pos;
                cell_entry.cell_id = split_entry.parent_cell_id * 4u + cell_pos;
                cell_entries[out_idx] = cell_entry;
            }
        }

        if (offset_idx == offsets_length - 1u) {
            result_info[0].cell_entries_length = curr_offset;
        }
    }
}
