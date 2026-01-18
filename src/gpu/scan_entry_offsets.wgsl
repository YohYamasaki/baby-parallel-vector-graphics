const WG_SIZE: u32 = 2u;

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

fn inclusive_scan_block(lid: u32) {
    var offset = 1u;
    loop {
        if (offset >= WG_SIZE) {
            break;
        }

        let curr = block_offsets[lid];
        var add = 0u;
        if (lid >= offset) {
            add = block_offsets[lid - offset];
        }

        workgroupBarrier();
        block_offsets[lid] = curr + add;
        workgroupBarrier();

        offset *= 2u;
    }
}

@group(0) @binding(0) var<storage, read_write> offsets_level_1: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets_level_2: array<u32>;

var<workgroup> block_offsets: array<u32, WG_SIZE>;

@compute
@workgroup_size(WG_SIZE)
fn scan_offset_block(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    let level_len = arrayLength(&offsets_level_1);
    let block_start = wg_linear * WG_SIZE;

    var block_len = 0u;
    if (block_start < level_len) {
        block_len = min(WG_SIZE, level_len - block_start);
    }
    if (block_len == 0u) {
        return;
    }

    if (idx < level_len) {
        block_offsets[lid.x] = offsets_level_1[idx];
    } else {
        block_offsets[lid.x] = 0u;
    }
    workgroupBarrier();

    inclusive_scan_block(lid.x);

    if (idx < level_len) {
        offsets_level_1[idx] = block_offsets[lid.x];
    }

    if (lid.x == 0u) {
        offsets_level_2[wg_linear] = block_offsets[block_len - 1u];
    }
}

@compute
@workgroup_size(WG_SIZE)
fn add_offset_carry(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    let level_len = arrayLength(&offsets_level_1);

    if (idx >= level_len || wg_linear == 0u) {
        return;
    }

    let carry_idx = wg_linear - 1u;
    if (carry_idx >= arrayLength(&offsets_level_2)) {
        return;
    }

    offsets_level_1[idx] += offsets_level_2[carry_idx];
}
