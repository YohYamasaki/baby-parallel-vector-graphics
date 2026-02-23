// Depends on: common.wgsl, split_helpers.wgsl

const WG_SIZE: u32 = 2u;

@group(0) @binding(0) var<storage, read_write> seg_entries: array<SegEntry>;
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

    let num_entries = result_info[0].seg_entries_length;
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
            var out_idx = curr_offset - lane_count;
            var seg_entry = SegEntry();

            if (emit_seg) {
                var shortcut = 0;
                if (has_up(split_info, cell_pos)) {
                    shortcut = 1;
                } else if (has_down(split_info, cell_pos)) {
                    shortcut = -1;
                }
                seg_entry.entry_type = ABSTRACT;
                seg_entry.data = shortcut;
                seg_entry.seg_idx = split_entry.seg_idx;
                seg_entry.path_idx = split_entry.path_idx;
                seg_entry.cell_pos = cell_pos;
                seg_entry.cell_id = split_entry.parent_cell_id * 4u + cell_pos;
                seg_entries[out_idx] = seg_entry;
                out_idx++;
            }

            if (emit_winc) {
                seg_entry.entry_type = WINDING_INCREMENT;
                seg_entry.data = winc;
                seg_entry.seg_idx = NONE_U32;
                seg_entry.path_idx = split_entry.path_idx;
                seg_entry.cell_pos = cell_pos;
                seg_entry.cell_id = split_entry.parent_cell_id * 4u + cell_pos;
                seg_entries[out_idx] = seg_entry;
            }
        }

        if (offset_idx == offsets_length - 1u) {
            result_info[0].seg_entries_length = curr_offset;
        }
    }
}
