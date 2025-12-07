use std::fs::File;
use std::io::BufWriter;

pub fn save_png_rgba8(path: &str, w: u32, h: u32, rgba: &[u8]) {
    let file = File::create(path).unwrap();
    let wtr = BufWriter::new(file);

    let mut encoder = png::Encoder::new(wtr, w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(rgba).unwrap();
}
