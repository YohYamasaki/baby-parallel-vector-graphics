use crate::abstract_segment::AbstractLineSegment;
use crate::cell_entry::{ABSTRACT, WINDING_INCREMENT};
use crate::path::{AbstractPath, Paint};
use crate::quad_tree::QuadTree;
use std::mem::swap;

const DRAW_DEBUG_OVERLAY: bool = true;

pub fn render_quadtree_by_node_array(
    tree: &QuadTree,
    abs_segments: &[AbstractLineSegment],
    abs_paths: &[AbstractPath],
    paints: &[Paint],
    pixels: &mut [u8],
    img_width: u32,
    img_height: u32,
) {
    for node in &tree.nodes {
        let Some(entry_range) = node.leaf_entry_range.as_ref() else {
            continue;
        };

        let left = node.bbox.left().max(0.0) as u32;
        let right = node.bbox.right().min(img_width as f32) as u32;
        let top = node.bbox.top().max(0.0) as u32;
        let bottom = node.bbox.bottom().min(img_height as f32) as u32;
        let line_paint = Paint::SolidColor { rgba: [255; 4] };

        for y in top..bottom {
            for x in left..right {
                let mut out = [0u8; 4];
                let mut has_shortcut = false;
                let mut winc = 0;
                let mut count = 0;
                for i in entry_range.start..entry_range.end {
                    let entry = &tree.entries[i];
                    let next_entry = if i == entry_range.end - 1 {
                        None
                    } else {
                        Some(&tree.entries[i + 1])
                    };
                    let is_segment = (entry.entry_type & ABSTRACT) != 0;
                    let is_winding_inc = (entry.entry_type & WINDING_INCREMENT) != 0;
                    if is_segment {
                        let seg = &abs_segments[entry.seg_idx as usize];
                        let [_, top, _, bottom] = seg.bbox_ltrb;
                        let shortcut = entry.data;

                        if seg.is_left(x as f32, y as f32)
                            && (y as f32) >= top
                            && (y as f32) < bottom
                        {
                            count += 1;
                        }

                        if shortcut != 0 && seg.hit_shortcut(&node.bbox, x as f32, y as f32) {
                            has_shortcut = true;
                            count += shortcut as i32;
                            // cb_left == 625.0 && cb_right == 750.0 && cb_top == 500.0 && seg_idx == 2 && x == 630 && y == 510
                        }
                    }

                    // No need to consider winding increment if there is no other abstract segment in the cell
                    // TODO: How to render a cell that does not have any segments but fully inside a path?
                    if is_winding_inc {
                        count += entry.data;
                        winc += entry.data;
                    }

                    let last_entry_in_path = next_entry
                        .is_some_and(|ne| ne.path_idx != entry.path_idx)
                        || next_entry.is_none();
                    if last_entry_in_path {
                        if count % 2 != 0 {
                            let path = &abs_paths[entry.path_idx as usize];
                            if let Paint::SolidColor { rgba } = paints[path.paint_id] {
                                out[..4].copy_from_slice(&rgba);
                            }
                        }
                        count = 0;
                    }
                }
                if DRAW_DEBUG_OVERLAY {
                    let debug_line_width = 6;
                    if has_shortcut && right - debug_line_width <= x && x <= right {
                        out[..4].copy_from_slice(&[0, 255, 0, 255]);
                    };
                    let mut curr = 8;
                    for _i in 0..winc.abs() as usize {
                        if winc != 0 && right - (curr + debug_line_width) <= x && x <= right - curr
                        {
                            if winc < 0 {
                                out[..4].copy_from_slice(&[255, 0, 0, 255]);
                            } else {
                                out[..4].copy_from_slice(&[0, 0, 255, 255]);
                            }
                        }
                        curr += debug_line_width + 6;
                    }
                }

                let base = ((y * img_width + x) * 4) as usize;
                pixels[base..base + 4].copy_from_slice(&out);
            }
        }

        if DRAW_DEBUG_OVERLAY {
            // QuadTree boxes
            draw_line(left, top, right - 1, top, pixels, &line_paint);
            draw_line(right - 1, top, right - 1, bottom - 1, pixels, &line_paint);
            draw_line(left, bottom - 1, right - 1, bottom - 1, pixels, &line_paint);
            draw_line(left, top, left, bottom - 1, pixels, &line_paint);
        }
    }
}

pub fn draw_line(x1: u32, y1: u32, x2: u32, y2: u32, pixels: &mut [u8], paint: &Paint) {
    let w = (x1 as i32 - x2 as i32).abs();
    let h = (y1 as i32 - y2 as i32).abs();
    let is_steep = w < h;
    let mut x1 = x1;
    let mut x2 = x2;
    let mut y1 = y1;
    let mut y2 = y2;
    if is_steep {
        swap(&mut x1, &mut y1);
        swap(&mut x2, &mut y2);
    }
    if x1 > x2 {
        swap(&mut x1, &mut x2);
        swap(&mut y1, &mut y2);
    }
    let mut y = y1 as f32;
    let step = (y2 as i32 - y1 as i32) as f32 / (x2 as i32 - x1 as i32) as f32;
    if let Paint::SolidColor { rgba } = paint {
        for x in x1..=x2 {
            let py = y.round() as u32;
            if is_steep {
                set_pixel(py, x, 1000, 1000, rgba, pixels);
            } else {
                set_pixel(x, py, 1000, 1000, rgba, pixels);
            }
            y = y + step;
        }
    }
}

fn set_pixel(x: u32, y: u32, width: u32, height: u32, rgba: &[u8; 4], pixels: &mut [u8]) {
    if x >= width || y >= height {
        return;
    }
    let base = ((y * width + x) * 4) as usize;
    pixels[base..base + 4].copy_from_slice(rgba);
}
