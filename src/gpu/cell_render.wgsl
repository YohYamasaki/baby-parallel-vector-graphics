const ABSTRACT: u32 = 1u << 0u;
const WINDING_INCREMENT: u32 = 1u << 3u;
const EPS: f32 = 1e-6;

struct CellMetadata {
    bbox_ltrb: vec4<f32>,
    mid: vec2<f32>,
    entry_start: u32,
    entry_count: u32,
    abstract_count: u32,
    _pad: array<u32, 3>,
}

struct SegEntry {
    entry_type: u32,
    data: i32,
    seg_idx: u32,
    path_idx: u32,
    cell_pos: u32,
    cell_id: u32,
    _pad: vec2<u32>,
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

struct PathPaintGpu {
    rgba: vec4<f32>,
}

struct RenderParams {
    width: u32,
    height: u32,
    entries_len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> cell_metadata: array<CellMetadata>;
@group(0) @binding(1) var<storage, read> seg_entries: array<SegEntry>;
@group(0) @binding(2) var<storage, read> segments: array<AbstractLineSegment>;
@group(0) @binding(3) var<storage, read> path_paints: array<PathPaintGpu>;
@group(0) @binding(4) var<uniform> params: RenderParams;
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba8unorm, write>;

fn contains_point(cell_meta: CellMetadata, px: u32, py: u32) -> bool {
    // Match CPU rasterization range conversion:
    // left/right/top/bottom are cast to u32 (truncate toward zero), then iterated as [left, right), [top, bottom).
    let l = u32(cell_meta.bbox_ltrb[0]);
    let t = u32(cell_meta.bbox_ltrb[1]);
    let r = u32(cell_meta.bbox_ltrb[2]);
    let b = u32(cell_meta.bbox_ltrb[3]);
    return px >= l && px < r && py >= t && py < b;
}

fn seg_eval(seg: AbstractLineSegment, x: f32, y: f32) -> f32 {
    return seg.a * x + seg.b * y + seg.c;
}

fn seg_is_left(seg: AbstractLineSegment, x: f32, y: f32) -> bool {
    return seg_eval(seg, x, y) < 0.0;
}

fn hit_shortcut(seg: AbstractLineSegment, cell_bbox_ltrb: vec4<f32>, sample_x: f32, sample_y: f32) -> bool {
    if (abs(seg.b) < EPS) {
        return false;
    }
    let x0 = cell_bbox_ltrb[2];
    let y0 = select(seg.y1, seg.y0, seg.x0 > seg.x1);
    if (sample_y >= y0) {
        return false;
    }
    return sample_x < x0;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let x = f32(gid.x);
    let y = f32(gid.y);
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    var i = 0u;
    let metadata_len = arrayLength(&cell_metadata);
    let path_paint_len = arrayLength(&path_paints);
    loop {
        if (i >= metadata_len) {
            break;
        }
        let cell_meta = cell_metadata[i];
        if (!(cell_meta.entry_count > 0u && contains_point(cell_meta, gid.x, gid.y))) {
            i += 1u;
            continue;
        }
        if (cell_meta.entry_start >= params.entries_len) {
            i += 1u;
            continue;
        }
        let remaining = params.entries_len - cell_meta.entry_start;
        if (cell_meta.entry_count > remaining) {
            i += 1u;
            continue;
        }
        let start = cell_meta.entry_start;
        let end = start + cell_meta.entry_count;
        var count = 0;
        var cell_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);

        var entry_idx = start;
        loop {
            if (entry_idx >= end) {
                break;
            }
            let entry = seg_entries[entry_idx];
            let is_segment = (entry.entry_type & ABSTRACT) != 0u;
            let is_winding_inc = (entry.entry_type & WINDING_INCREMENT) != 0u;

            if (is_segment) {
                let seg = segments[entry.seg_idx];
                let top = seg.bbox_ltrb[1];
                let bottom = seg.bbox_ltrb[3];
                if (seg_is_left(seg, x, y) && y >= top && y < bottom) {
                    count += 1;
                }
                if (entry.data != 0 && hit_shortcut(seg, cell_meta.bbox_ltrb, x, y)) {
                    count += entry.data;
                }
            }

            if (is_winding_inc) {
                count += entry.data;
            }

            var last_entry_in_path = entry_idx + 1u >= end;
            if (!last_entry_in_path) {
                last_entry_in_path = seg_entries[entry_idx + 1u].path_idx != entry.path_idx;
            }
            if (last_entry_in_path) {
                if ((count & 1) != 0 && path_paint_len > 0u) {
                    let paint_idx = min(entry.path_idx, path_paint_len - 1u);
                    cell_color = path_paints[paint_idx].rgba;
                }
                count = 0;
            }

            entry_idx += 1u;
        }

        color = cell_color;
        i = i + 1u;
    }

    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
