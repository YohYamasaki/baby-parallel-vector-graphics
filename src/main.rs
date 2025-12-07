use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let svg = fs::read_to_string("src/sample/simple_rects.svg")?;
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_str(&svg, &opt)?;
    println!("{}", tree.size().height());
    Ok(())
}
