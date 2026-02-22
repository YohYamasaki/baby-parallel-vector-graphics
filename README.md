# baby-parallel-vector-graphics

A very much in-progress Rust + wgpu project implementing the paper [Massively-Parallel Vector
Graphics](https://w3.impa.br/~diego/projects/GanEtAl14/).

This repository is primarily a personal learning project focused on understanding how a tile-based 2D vector
renderer can be built on GPU compute pipelines.

## Current state

- Only line segments are supported for geometry.
- Paths are treated as fillable polygons made of `MoveTo`/`LineTo`/`Close`.
- Fill-only rendering (no stroke pipeline yet).
- Fill rule is currently even-odd in practice.
- Quadratic and cubic segments are not implemented.
- Anti-aliasing is not implemented.
- Almost no performance / memory optimisation.

## High-Level Pipeline

1. Parse SVG into abstract line segments and path metadata.
2. Build root seg entries.
3. Subdivide cells on GPU (quadtree-style) and produce `CellMetadata` + `SegEntry` arrays.
4. Render on GPU compute (`src/gpu/cell_render.wgsl`) into an offscreen texture.
5. Read back GPU output and save as PNG.
6. Also render with CPU reference path and save PNG for comparison.

## How To Run

```bash
cargo run
```

Outputs:

- `output/test_gpu.png`
- `output/test_cpu.png`

Input SVG is currently loaded from:

- `sample_svg/simple_polygons.svg`
