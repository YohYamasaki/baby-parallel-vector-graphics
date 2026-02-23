// Depends on: common.wgsl

const WG_SIZE: u32 = 1u;

@group(0) @binding(0) var<storage, read> seg_entries: array<SegEntry>;
@group(0) @binding(1) var<storage, read_write> cell_metadata: array<CellMetadata>;
@group(0) @binding(2) var<storage, read_write> result_info: array<SplitResultInfo>;

@compute
@workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let entry_idx = linearize_workgroup_id(wid, num_wg);
    let result_len = result_info[0].seg_entries_length;

    if (entry_idx >= result_len) {
        return;
    }

    let entry = seg_entries[entry_idx];
    let cell_id = entry.cell_id;

    if (entry_idx == 0 || seg_entries[entry_idx - 1].cell_id != cell_id) {
        cell_metadata[cell_id].entry_start = entry_idx;

        var end = entry_idx + 1u;
        var abs_count = select(0u, 1u, (entry.entry_type & ABSTRACT) != 0u);
        while (end < result_len && seg_entries[end].cell_id == cell_id) {
            abs_count += select(0u, 1u, (seg_entries[end].entry_type & ABSTRACT) != 0u);
            end++;
        }
        cell_metadata[cell_id].entry_count = end - entry_idx;
        cell_metadata[cell_id].abstract_count = abs_count;
    }
}
