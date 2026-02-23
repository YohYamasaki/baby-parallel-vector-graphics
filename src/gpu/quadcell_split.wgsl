// Depends on: common.wgsl

const WG_SIZE: u32 = 1u;

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

@group(0) @binding(0) var<storage, read> cell_metadata_in: array<CellMetadata>;
@group(0) @binding(1) var<storage, read_write> cell_metadata_out: array<CellMetadata>;
// result_info provides min_seg threshold for split skipping.
@group(0) @binding(2) var<storage, read> result_info: array<SplitResultInfo>;

@compute
@workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let cell_id = linearize_workgroup_id(wid, num_wg);
    let metadata = cell_metadata_in[cell_id];
    let parent_bbox = metadata.bbox_ltrb;
    let min_seg = result_info[0].min_seg;

    // Use abstract_count (segments only) for leaf determination; WINDING_INCREMENT entries
    // must not prevent a cell from being treated as a leaf.
    let is_leaf = metadata.abstract_count <= min_seg;

    let mid_x = (parent_bbox[2] + parent_bbox[0]) / 2;
    let mid_y = (parent_bbox[3] + parent_bbox[1]) / 2;
    let child_bounds = get_child_bounds(parent_bbox, mid_x, mid_y);
    let base = cell_id * 4u;

    for (var i = 0u; i < 4u; i++) {
        var child_meta: CellMetadata;
        child_meta.bbox_ltrb = child_bounds[i];
        let cb = child_bounds[i];
        child_meta.mid = vec2((cb[0] + cb[2]) * 0.5, (cb[1] + cb[3]) * 0.5);
        // entry_start and entry_count are written by quadcell_update_metadata after emission.
        child_meta.entry_start = 0u;
        child_meta.entry_count = 0u;
        cell_metadata_out[base + i] = child_meta;
    }
}
