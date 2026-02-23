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
