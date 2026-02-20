const WG_SIZE: u32 = 2u;

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
}

struct WindingBlockInfo {
    first_path_idx: u32,
    last_path_idx: u32,
    first_cell_id: u32,
    last_cell_id: u32,
    tail_winding: vec4<i32>,
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
    parent_cell_id: u32, // Propagated from entry.cell_id to track parent cell across subdivision
}

struct EdgeIntersectionInfo {
    /*     TL ---10 --- T ---11 --- TR -- 14 -- TI
           |            |           |
           7            8           9
           |            |           |
           L  --- 5 --- C --- 6 --- R  -- 13 -- I
           |            |           |
           2            3           4
           |            |           |
           BL --- 0 --- B --- 1 --- BR -- 12 -- BI

           11 and 14 -> 15
            6 and 13 -> 16
            1 and 12 -> 17
    */
    cross0: bool,
    cross1: bool,
    cross2: bool,
    cross3: bool,

    cross4: bool,
    cross5: bool,
    cross6: bool,
    cross7: bool,

    cross8: bool,
    cross9: bool,
    cross10: bool,
    cross11: bool,

    cross12: bool,
    cross13: bool,
    cross14: bool,
    cross15: bool,

    cross16: bool,
    cross17: bool,
}

fn linearize_workgroup_id(wid: vec3<u32>, num_wg: vec3<u32>) -> u32 {
    // linear = x + y*X + z*(X*Y)
    return wid.x + wid.y * num_wg.x + wid.z * (num_wg.x * num_wg.y);
}

fn going_right(direction: u32) -> bool {
    switch (direction) {
        case 0: { return false; } // NW
        case 1: { return true; } // NE
        case 2: { return false; } // SW
        case 3: { return true; } // SE
        case 4: { return true; } // Horizontal
        default: { return false; }
    }
}

fn going_up(direction: u32) -> bool {
    switch (direction) {
        case 0: { return true; } // NW
        case 1: { return true; } // NE
        case 2: { return false; } // SW
        case 3: { return false; } // SE
        case 4: { return false; } // Horizontal
        default: { return false; }
    }
}

fn flag(x: u32, offset: u32) -> u32 {
    return (1u << x) << offset;
}

fn fill(cell: u32) -> u32 {
    return flag(cell, 0);
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

fn contains_point_in_bbox(x: f32, y: f32, bbox_ltrb: vec4<f32>) -> bool {
    let left = bbox_ltrb[0];
    let top = bbox_ltrb[1];
    let right = bbox_ltrb[2];
    let bottom = bbox_ltrb[3];
    return x >= left && x <= right && y >= top && y <= bottom;
}

fn classify_child(x: f32, y: f32, mid_x: f32, mid_y: f32) -> u32 {
    if (x <= mid_x) {
        if (y <= mid_y) {
            return TOP_LEFT;
        }
        return BOTTOM_LEFT;
    }
    if (y <= mid_y) {
        return TOP_RIGHT;
    }
    return BOTTOM_RIGHT;
}

fn hit_chull() -> i32 {
    return -1; // TODO: this is only for line segment
}

fn eval(a: f32, b: f32, c: f32, x: f32, y: f32) -> f32 {
    return a * x + b * y + c;
}

// TODO: probably better to do not pass entire AbstractLineSegment
fn half_open_eval(seg: AbstractLineSegment, sample_x: f32, sample_y: f32) -> i32 {
    let left = seg.bbox_ltrb[0];
    let top = seg.bbox_ltrb[1];
    let right = seg.bbox_ltrb[2];
    let bottom = seg.bbox_ltrb[3];

    // Outside vertical bbox range?
    if sample_y > bottom || sample_y <= top {
        // Only meaningful if x is inside bbox range; otherwise 0.
        if !(left <= sample_x && sample_x < right) {
            return 0;
        }

        let same_dir = going_right(seg.direction) == going_up(seg.direction);

        // Above or below decides which sign to return.
        // This early return is for avoiding running area evaluation by the implicit function of the segment,
        // but clipping the value
        if sample_y <= top {
            return select(1, -1, same_dir);
        } else {
            return select(-1, 1, same_dir);
        };
    }

    // Within vertical range: classify by x against bbox.
    if sample_x >= right {
        return 1;
    }
    if sample_x < left {
        return -1;
    }

    // Within bbox: optional hull check.
    let check = hit_chull();
    if check != -1 {
        return select(1, -1, check == 1);
    }

    // Fallback to implicit evaluation sign.
    if eval(seg.a, seg.b, seg.c, sample_x, sample_y) < 0. {
        return -1;
    }
    return 1;
}

fn get_edge_intersection_info(seg: AbstractLineSegment, bbox_ltrb: vec4<f32>, mid_x: f32, mid_y: f32) -> EdgeIntersectionInfo {
    let left_bound = bbox_ltrb[0];
    let top_bound = bbox_ltrb[1];
    let right_bound = bbox_ltrb[2];
    let bottom_bound = bbox_ltrb[3];
    let width = right_bound - left_bound;
    let far_x = right_bound + (width + 1.0) * 1024.0; // TODO: 無限遠のレイ用の関数を作る

    let sign_l = half_open_eval(seg, left_bound, mid_y);
    let sign_c = half_open_eval(seg, mid_x, mid_y);
    let sign_r = half_open_eval(seg, right_bound, mid_y);
    let sign_i = half_open_eval(seg, far_x, mid_y);
    let sign_bl = half_open_eval(seg, left_bound, bottom_bound);
    let sign_b = half_open_eval(seg, mid_x, bottom_bound);
    let sign_br = half_open_eval(seg, right_bound, bottom_bound);
    let sign_bi = half_open_eval(seg, far_x, bottom_bound);
    let sign_tl = half_open_eval(seg, left_bound, top_bound);
    let sign_t = half_open_eval(seg, mid_x, top_bound);
    let sign_tr = half_open_eval(seg, right_bound, top_bound);
    let sign_ti = half_open_eval(seg, far_x, top_bound);

    var einfo = EdgeIntersectionInfo();
    einfo.cross0 = sign_bl * sign_b < 0;
    einfo.cross1 = sign_b * sign_br < 0;
    einfo.cross2 = sign_bl * sign_l < 0;
    einfo.cross3 = sign_b * sign_c < 0;
    einfo.cross4 = sign_br * sign_r < 0;
    einfo.cross5 = sign_l * sign_c < 0;
    einfo.cross6 = sign_c * sign_r < 0;
    einfo.cross7 = sign_tl * sign_l < 0;
    einfo.cross8 = sign_t * sign_c < 0;
    einfo.cross9 = sign_tr * sign_r < 0;
    einfo.cross10 = sign_tl * sign_t < 0;
    einfo.cross11 = sign_t * sign_tr < 0;
    einfo.cross12 = sign_br * sign_bi < 0;
    einfo.cross13 = sign_r * sign_i < 0;
    einfo.cross14 = sign_tr * sign_ti < 0;
    einfo.cross15 = sign_t * sign_ti < 0;
    einfo.cross16 = sign_c * sign_i < 0;
    einfo.cross17 = sign_b * sign_bi < 0;
    return einfo;
}

fn build_split_data(
    seg: AbstractLineSegment,
    shortcut: i32,
    einfo: EdgeIntersectionInfo,
    bbox_ltrb: vec4<f32>,
    mid_x: f32,
    mid_y: f32) -> SplitData {
    // TODO: should we do split contain check in here? We can do it outside of this fn
    // we can use AbstractLineSegment::is_inside_bb
    // TODO: probably??
    let going_up = select(-1, 1, seg.y0 > seg.y1);
    let going_right = select(-1, 1, seg.x0 < seg.x1);
    var split_info = 0u;
    var winding = vec4(0, 0, 0, 0);

    // endpoints inside parent quad must mark child occupancy.
    if (contains_point_in_bbox(seg.x0, seg.y0, bbox_ltrb)) {
        split_info |= fill(classify_child(seg.x0, seg.y0, mid_x, mid_y));
    }
    if (contains_point_in_bbox(seg.x1, seg.y1, bbox_ltrb)) {
        split_info |= fill(classify_child(seg.x1, seg.y1, mid_x, mid_y));
    }

    if einfo.cross0 {
        split_info |= fill(BOTTOM_LEFT);
    }

    if einfo.cross1 {
        split_info |= fill(BOTTOM_RIGHT);
        winding[BOTTOM_LEFT] += going_up;
    }

    if einfo.cross2 {
        split_info |= fill(BOTTOM_LEFT);
    }

    if einfo.cross3 {
        split_info |= fill(BOTTOM_LEFT) | fill(BOTTOM_RIGHT);
        if !einfo.cross16 {
            if !einfo.cross17 {
                if going_right > 0 {
                    split_info |= up(BOTTOM_LEFT);
                } else {
                    split_info |= down(BOTTOM_LEFT);
                }
            } else {
                winding[BOTTOM_LEFT] += going_right;
            }
        }
    }

    if einfo.cross4 {
        split_info |= fill(BOTTOM_RIGHT);
        if !einfo.cross13 {
            if !einfo.cross12 {
                if going_right > 0 {
                    split_info |= up(BOTTOM_RIGHT);
                } else {
                    split_info |= down(BOTTOM_RIGHT);
                }
            } else {
                winding[BOTTOM_RIGHT] += going_right;
            }
        }
    }

    if einfo.cross5 {
        split_info |= fill(BOTTOM_LEFT) | fill(TOP_LEFT);
    }

    if einfo.cross6 {
        split_info |= fill(BOTTOM_RIGHT) | fill(TOP_RIGHT);
        winding[TOP_LEFT] += going_up;
    }

    if einfo.cross7 {
        split_info |= fill(TOP_LEFT);
    }

    if einfo.cross8 {
        split_info |= fill(TOP_LEFT) | fill(TOP_RIGHT);
        if !einfo.cross15 {
            if !einfo.cross16 {
                if going_right > 0 {
                    split_info |= up(TOP_LEFT);
                } else {
                    split_info |= down(TOP_LEFT);
                }
            } else {
                winding[TOP_LEFT] += going_right;
            }
        }
    }

    if einfo.cross9 {
        split_info |= fill(TOP_RIGHT);
        if !einfo.cross14 {
            if !einfo.cross13 {
                if going_right > 0 {
                    split_info |= up(TOP_RIGHT);
                } else {
                    split_info |= down(TOP_RIGHT);
                }
            } else {
                winding[TOP_RIGHT] += going_right;
            }
        }
    }

    if einfo.cross10 {
        split_info |= fill(TOP_LEFT);
    }

    if einfo.cross11 {
        split_info |= fill(TOP_RIGHT);
    }

    if einfo.cross12 {
        winding[BOTTOM_RIGHT] += going_up;
        winding[BOTTOM_LEFT] += going_up;
    }

    if einfo.cross13 {
        winding[TOP_RIGHT] += going_up;
        winding[TOP_LEFT] += going_up;
    }



    if shortcut != 0 {
        let is_p0_shortcut_base = seg.x0 > seg.x1;
        let shortcut_base_x = select(seg.x1, seg.x0, is_p0_shortcut_base);
        let shortcut_base_y = select(seg.y1, seg.y0, is_p0_shortcut_base);
        let is_down_shortcut = shortcut == -1;

        let left_bound = bbox_ltrb[0];
        let top_bound = bbox_ltrb[1];
        let right_bound = bbox_ltrb[2];
        let bottom_bound = bbox_ltrb[3];

        if !(shortcut_base_y <= top_bound || shortcut_base_x < left_bound) &&
             shortcut_base_x >= right_bound && shortcut_base_y >= mid_y {
             let winc = select(1, -1, is_down_shortcut);
            winding[TOP_LEFT] += winc;
            winding[TOP_RIGHT] += winc;

            if shortcut_base_y >= bottom_bound {
                winding[BOTTOM_LEFT] += winc;
                winding[BOTTOM_RIGHT] += winc;
            }
        }
    }

    var split_data = SplitData();
    split_data.winding = winding;
    split_data.split_info = split_info;
    return split_data;
}

/// Kernel 1 of 4.2 Parallel subdivision
/// Assuming parent_entries already ordered SEGMENTs - WINDING for each cell.
fn build_split_entries(idx: u32) {
    // Read the actual entry count written by process_level() before this dispatch.
    let n = result_info[0].cell_entries_length;
    let entry = cell_entries[idx];
    let metadata = cell_metadata[entry.cell_id];
    let is_abstract_entry = (entry.entry_type & ABSTRACT) != 0;
    let is_winding_inc_entry = (entry.entry_type & WINDING_INCREMENT) != 0;

    if is_abstract_entry {
        let seg_idx = entry.seg_idx;
        let seg = segments[seg_idx];
        let edge_info = get_edge_intersection_info(
            seg,
            metadata.bbox_ltrb,
     metadata.mid[0],
     metadata.mid[1]
        );
        let split_data = build_split_data(
            seg,
            entry.data,
            edge_info,
            metadata.bbox_ltrb,
     metadata.mid[0],
     metadata.mid[1]
        );

        // Add offsets (if a child cell intersects with the segment, add 1 to the offset)
        global_cell_offsets[TOP_LEFT * n + idx] = has_fill(split_data.split_info, TOP_LEFT);
        global_cell_offsets[TOP_RIGHT * n + idx] = has_fill(split_data.split_info, TOP_RIGHT);
        global_cell_offsets[BOTTOM_LEFT * n + idx] = has_fill(split_data.split_info, BOTTOM_LEFT);
        global_cell_offsets[BOTTOM_RIGHT * n + idx] = has_fill(split_data.split_info, BOTTOM_RIGHT);

        var split_entry = SplitEntry();
        split_entry.split_data = split_data;
        split_entry.unique_id = idx;
        split_entry.seg_idx = seg_idx;
        split_entry.path_idx = seg.path_idx;
        split_entry.parent_cell_id = entry.cell_id;

        global_split_entries[idx] = split_entry;

        // Keep a per-entry winding representation so block scan can start from all entries.
        var winfo = WindingBlockInfo();
        winfo.first_path_idx = split_entry.path_idx;
        winfo.last_path_idx = split_entry.path_idx;
        winfo.first_cell_id = entry.cell_id;
        winfo.last_cell_id = entry.cell_id;
        winfo.tail_winding = split_data.winding;
        winding_infos[idx] = winfo;
    }

    // TODO: not sure about the winding increment entry handling is correct
    if is_winding_inc_entry {
        let parent_winding = entry.data;

        var split_data = SplitData();
        split_data.winding = vec4(parent_winding, parent_winding, parent_winding, parent_winding);
        split_data.split_info = 0;

        var split_entry = SplitEntry();
        split_entry.split_data = split_data;
        split_entry.unique_id = idx;
        split_entry.seg_idx = NONE_U32;
        split_entry.path_idx = entry.path_idx;
        split_entry.parent_cell_id = entry.cell_id;

        global_split_entries[idx] = split_entry;
        global_cell_offsets[TOP_LEFT * n + idx] = 0u;
        global_cell_offsets[TOP_RIGHT * n + idx] = 0u;
        global_cell_offsets[BOTTOM_LEFT * n + idx] = 0u;
        global_cell_offsets[BOTTOM_RIGHT * n + idx] = 0u;

        // Keep the same representation for winding increment entries.
        var winfo = WindingBlockInfo();
        winfo.first_path_idx = entry.path_idx;
        winfo.last_path_idx = entry.path_idx;
        winfo.first_cell_id = entry.cell_id;
        winfo.last_cell_id = entry.cell_id;
        winfo.tail_winding = split_data.winding;
        winding_infos[idx] = winfo;
    }
}


struct SplitResultInfo {
    cell_entries_length: u32,
}

@group(0) @binding(0) var<storage, read_write> cell_entries: array<CellEntry>;
@group(0) @binding(1) var<storage, read> segments: array<AbstractLineSegment>;
@group(0) @binding(2) var<storage, read> cell_metadata: array<CellMetadata>;
@group(0) @binding(3) var<storage, read_write> global_split_entries: array<SplitEntry>;
@group(0) @binding(4) var<storage, read_write> global_cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> winding_infos: array<WindingBlockInfo>;
// result_info[0].cell_entries_length holds the actual number of entries for the current depth,
// written by process_level() before this shader runs.
@group(0) @binding(6) var<storage, read_write> result_info: array<SplitResultInfo>;

@compute
@workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let wg_linear = linearize_workgroup_id(wid, num_wg);
    let idx = wg_linear * WG_SIZE + lid.x;
    // Read the actual entry count for the current depth from result_info.
    let entries_length = result_info[0].cell_entries_length;
    let in_range = idx < entries_length;

    if (in_range) {
        build_split_entries(idx);
    }
}
