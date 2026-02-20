use crate::abstract_segment::AbstractLineSegment;
use crate::cell_entry::{print_split_entries, CellEntry, SplitEntry};
use crate::geometry::rect::Rect;
use crate::gpu::init::init_wgpu;
use crate::gpu::quad_tree::CellMetadata;
use bytemuck::{bytes_of, AnyBitPattern, Pod, Zeroable};
use std::sync::mpsc::channel;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::BufferDescriptor;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry, BindingType, Buffer,
    BufferBindingType, BufferUsages, ComputePipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

const WG_SIZE: u32 = 2;

fn split_dispatch_3d(workgroups_needed: u32, max_dim: u32) -> [u32; 3] {
    let x = workgroups_needed.min(max_dim).max(1);
    let remaining_after_x = (workgroups_needed + x - 1) / x;
    let y = remaining_after_x.min(max_dim);

    let xy = (x as u64) * (y as u64);
    let z = ((workgroups_needed as u64) + xy - 1) / xy;
    assert!(z <= max_dim as u64, "dispatch exceeds max_dim^3");

    [x, y, z as u32]
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct SplitResultInfo {
    pub cell_entries_length: u32,
    // Minimum entry count threshold for splitting a cell further.
    // Cells with entry_count <= min_seg are treated as leaves in quadcell_split.wgsl.
    pub min_seg: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ScanParams {
    level_len: u32,
    carry_len: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct WindingBlockInfo {
    first_path_idx: u32,
    last_path_idx: u32,
    all_same_path: u32,
    _pad: u32,

    tail_winding: [i32; 4],
}

struct Resources {
    // metadata (ping-pong: depth % 2 selects input/output)
    cell_metadata_buffer_1: wgpu::Buffer,
    cell_metadata_buffer_2: wgpu::Buffer,
    // cell entries: single buffer, Kernel 1 reads then Kernel 5 overwrites in-place
    cell_entries_buffer: wgpu::Buffer,
    segments_buffer: wgpu::Buffer,
    // intermediates
    split_entries_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer,
    winding_block_sum_buffers: Vec<Buffer>,
    winding_scan_params_buffers: Vec<Buffer>,
    offset_block_sum_buffers: Vec<Buffer>,
    offset_scan_params_buffers: Vec<Buffer>,
    // result info
    result_info_buffer: wgpu::Buffer,
    // readbacks
    winding_block_sum_readback_buffers: Vec<Buffer>,
    split_entries_readback_buffer: wgpu::Buffer,
    cell_offsets_readback_buffer: wgpu::Buffer,
    cell_metadata_readback_buffer: wgpu::Buffer,
    cell_entry_readback_buffer: wgpu::Buffer,
    result_info_readback_buffer: wgpu::Buffer,
}

impl Resources {
    fn new(
        device: &wgpu::Device,
        cell_entries: &[CellEntry],
        segments: &[AbstractLineSegment],
        max_depth: u8,
    ) -> Self {
        let limits = device.limits();
        let max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size as u64;
        let max_buffer_size = limits.max_buffer_size;
        let check_storage_size = |label: &str, bytes: u64| {
            assert!(
                bytes <= max_storage_buffer_binding_size,
                "{label} size {bytes} exceeds max_storage_buffer_binding_size {max_storage_buffer_binding_size}"
            );
            assert!(
                bytes <= max_buffer_size,
                "{label} size {bytes} exceeds max_buffer_size {max_buffer_size}"
            );
            bytes
        };
        let checked_pow4 = |exp: u8| -> u64 {
            let mut out = 1u64;
            for _ in 0..exp {
                out = out
                    .checked_mul(4)
                    .expect("entry capacity overflow while computing 4^max_depth");
            }
            out
        };

        let initial_entries = cell_entries.len().max(1) as u64;
        let max_cell_entries = initial_entries
            .checked_mul(checked_pow4(max_depth))
            .expect("max_cell_entries overflow")
            .max(1);
        let max_split_entries = if max_depth == 0 {
            initial_entries
        } else {
            initial_entries
                .checked_mul(checked_pow4(max_depth - 1))
                .expect("max_split_entries overflow")
        }
        .max(1);
        let max_offsets = max_split_entries
            .checked_mul(4)
            .expect("max_offsets overflow")
            .max(1);

        // Single cell entries buffer: Kernel 1 finishes reading before Kernel 5 writes,
        // so in-place overwrite is safe across dispatches.
        let cell_entries_buf_size = check_storage_size(
            "cell entries buffer",
            max_cell_entries
                .checked_mul(size_of::<CellEntry>() as u64)
                .expect("cell entries buffer size overflow"),
        );
        let cell_entries_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cell entries buffer"),
            size: cell_entries_buf_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_metadata_buf_size = check_storage_size(
            "cell metadata buffer",
            checked_pow4(max_depth)
                .checked_mul(size_of::<CellMetadata>() as u64)
                .expect("cell metadata buffer size overflow")
                .max(size_of::<CellMetadata>() as u64),
        );
        let create_metadata_buffer = |label: &str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: cell_metadata_buf_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cell_metadata_buffer_1 = create_metadata_buffer("cell metadata buffer 1");
        let cell_metadata_buffer_2 = create_metadata_buffer("cell metadata buffer 2");
        let segments_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("segments buffer"),
            contents: bytemuck::cast_slice(segments),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        // Intermediate structures are sized for the requested max_depth.
        // max_split_entries: per-level input capacity.
        // max_cell_entries: per-level output capacity (4x split entries).
        let split_entries_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("split entries buffer"),
            size: check_storage_size(
                "split entries buffer",
                max_split_entries
                    .checked_mul(size_of::<SplitEntry>() as u64)
                    .expect("split entries buffer size overflow"),
            ),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // cell_offsets: 4 interleaved arrays, each of length max_split_entries.
        let cell_offsets_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cell offsets buffer"),
            size: check_storage_size(
                "cell offsets buffer",
                max_offsets
                    .checked_mul(size_of::<u32>() as u64)
                    .expect("cell offsets buffer size overflow")
                    .max(size_of::<u32>() as u64),
            ),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create buffers for each level since the prefix sum process is done recursively the following process.
        // Scan by blocks using Hillis-Steele -> add carry from the one previous block's last element
        let create_sum_buffer = |bytes: u64| {
            let checked = check_storage_size("winding block sum buffer", bytes.max(32));
            device.create_buffer(&BufferDescriptor {
                label: Some("winding block sum buffer"),
                size: checked,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        // Winding block sum buffers sized for max_split_entries.
        let cell_entries_bytes = max_split_entries
            .checked_mul(size_of::<WindingBlockInfo>() as u64)
            .expect("winding block sum level-0 size overflow");
        let mut winding_block_sum_buffers: Vec<Buffer> =
            vec![create_sum_buffer(cell_entries_bytes)];
        let mut level_elms = max_split_entries as usize;
        while level_elms > WG_SIZE as usize {
            let num_blocks = level_elms.div_ceil(WG_SIZE as usize).max(1);
            let bytes = (num_blocks * size_of::<WindingBlockInfo>()) as u64;
            winding_block_sum_buffers.push(create_sum_buffer(bytes));
            level_elms = num_blocks;
        }
        // Create sentinel buffer for the last block sum that does not require more splitting,
        // But we don't want to create another buffer, pipeline, and bindgroup only for it
        winding_block_sum_buffers.push(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("winding block sum sentinel buffer"),
            contents: bytes_of(&[0u32; 8]), // minimum bytes of the buffer is 32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        }));

        // Hierarchical scan buffers for offsets.
        // Level 0 uses `cell_offsets_buffer`; this vector keeps level>=1 and a sentinel.
        let create_offset_sum_buffer = |bytes: u64| {
            let checked = check_storage_size("offset block sum buffer", bytes.max(size_of::<u32>() as u64));
            device.create_buffer(&BufferDescriptor {
                label: Some("offset block sum buffer"),
                size: checked,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        // Offset block sum buffers: max_offsets elements.
        let mut offset_block_sum_buffers: Vec<Buffer> = vec![];
        let mut offset_level_elms = max_offsets as usize;
        while offset_level_elms > WG_SIZE as usize {
            let num_blocks = offset_level_elms.div_ceil(WG_SIZE as usize).max(1);
            let bytes = (num_blocks * size_of::<u32>()) as u64;
            offset_block_sum_buffers.push(create_offset_sum_buffer(bytes));
            offset_level_elms = num_blocks;
        }
        // Sentinel for the top-level carry source.
        offset_block_sum_buffers.push(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("offset block sum sentinel buffer"),
            contents: bytes_of(&[0u32; 1]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        }));

        let create_scan_params_buffer = |label: &str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: check_storage_size("scan params buffer", size_of::<ScanParams>() as u64),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let winding_scan_params_buffers = (0..winding_block_sum_buffers.len().saturating_sub(1))
            .map(|_| create_scan_params_buffer("winding scan params buffer"))
            .collect();
        let offset_scan_params_buffers =
            (0..(1 + offset_block_sum_buffers.len()).saturating_sub(1))
                .map(|_| create_scan_params_buffer("offset scan params buffer"))
                .collect();

        let result_info_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("result info buffer"),
            size: size_of::<SplitResultInfo>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // readback buffers
        let result_entries_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug out buffer"),
            size: cell_entries_buf_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let split_entries_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("parent bounds buffer"),
            size: split_entries_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let result_info_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("result info readback buffer"),
            size: result_info_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let cell_metadata_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cell metadata readback buffer"),
            size: cell_metadata_buffer_1.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let cell_offsets_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("cell offsets readback buffer"),
            size: cell_offsets_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let winding_block_sum_readback_buffers = winding_block_sum_buffers
            .iter()
            .enumerate()
            .map(|(level, buffer)| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("winding block sum readback buffer level {level}")),
                    size: buffer.size(),
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            // cell metadata
            cell_metadata_buffer_2,
            cell_metadata_buffer_1,
            // cell entries (single buffer, in-place overwrite)
            cell_entries_buffer,
            segments_buffer,
            // intermediates
            split_entries_buffer,
            cell_offsets_buffer,
            winding_block_sum_buffers,
            winding_scan_params_buffers,
            offset_block_sum_buffers,
            offset_scan_params_buffers,
            // result info
            result_info_buffer,
            // readbacks
            winding_block_sum_readback_buffers,
            cell_offsets_readback_buffer,
            split_entries_readback_buffer,
            result_info_readback_buffer,
            cell_metadata_readback_buffer,
            cell_entry_readback_buffer: result_entries_readback_buffer,
        }
    }
}

struct Pipelines {
    quadcell_split: wgpu::ComputePipeline,
    build_split_entries: wgpu::ComputePipeline,
    scan_winding_block: wgpu::ComputePipeline,
    scan_offset_block: wgpu::ComputePipeline,
    add_offset_carry: wgpu::ComputePipeline,
    emit_cell_entries: wgpu::ComputePipeline,
    mark_tail_winding_offsets: wgpu::ComputePipeline,
    add_winding_carry: wgpu::ComputePipeline,
    update_metadata: wgpu::ComputePipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let quadcell_split_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("quadcell split shader"),
            source: ShaderSource::Wgsl(include_str!("quadcell_split.wgsl").into()),
        });
        let split_cell_entry_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("split shader"),
            source: ShaderSource::Wgsl(include_str!("build_split_entries.wgsl").into()),
        });
        let scan_winding_block_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("scan winding block shader"),
            source: ShaderSource::Wgsl(include_str!("winding_block_sum.wgsl").into()),
        });
        let scan_entry_offsets_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("scan entry offsets shader"),
            source: ShaderSource::Wgsl(include_str!("scan_entry_offsets.wgsl").into()),
        });
        let split_to_cell_entry_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("split to cell entry shader"),
            source: ShaderSource::Wgsl(include_str!("split_to_cell_entry.wgsl").into()),
        });
        let update_metadata_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("update metadata shader"),
            source: ShaderSource::Wgsl(include_str!("quadcell_update_metadata.wgsl").into()),
        });

        let quadcell_split = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("quadcell split pipeline"),
            layout: None,
            module: &quadcell_split_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let bgl_storage_entry = |binding: u32| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // winding shaders:
        // bindings 0-4 (cell_entries, split_entries, cell_offsets, winding_1, winding_2)
        // binding 5 (result_info), binding 6 (per-dispatch scan params)
        let winding_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("winding bind group"),
            entries: &[
                bgl_storage_entry(0),
                bgl_storage_entry(1),
                bgl_storage_entry(2),
                bgl_storage_entry(3),
                bgl_storage_entry(4),
                bgl_storage_entry(5),
                bgl_storage_entry(6),
            ],
        });

        let winding_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("winding pl"),
            bind_group_layouts: &[&winding_bgl],
            immediate_size: 0,
        });

        let offset_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("offset bind group"),
            entries: &[
                bgl_storage_entry(0),
                bgl_storage_entry(1),
                bgl_storage_entry(2),
            ],
        });
        let offset_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("offset pl"),
            bind_group_layouts: &[&offset_bgl],
            immediate_size: 0,
        });

        let build_split = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("split pipeline"),
            layout: None,
            module: &split_cell_entry_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        let scan_winding_block = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("scan winding block pipeline"),
            layout: Some(&winding_pl),
            module: &scan_winding_block_shader,
            entry_point: Some("scan_winding_block"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let add_winding_carry = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("add winding carry pipeline"),
            layout: Some(&winding_pl),
            module: &scan_winding_block_shader,
            entry_point: Some("add_winding_carry"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        let mark_tail_winding_offsets =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("mark tail winding pipeline"),
                layout: Some(&winding_pl),
                module: &scan_winding_block_shader,
                entry_point: Some("mark_tail_winding"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        let scan_offset_block = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("offsets block scan pipeline"),
            layout: Some(&offset_pl),
            module: &scan_entry_offsets_shader,
            entry_point: Some("scan_offset_block"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        let add_offset_carry = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("add offsets carry pipeline"),
            layout: Some(&offset_pl),
            module: &scan_entry_offsets_shader,
            entry_point: Some("add_offset_carry"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        let emit_cell_entries = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("split to cell pipeline"),
            layout: None,
            module: &split_to_cell_entry_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        let update_metadata = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("update metadata pipeline"),
            layout: None,
            module: &update_metadata_shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });
        Self {
            quadcell_split,
            build_split_entries: build_split,
            scan_winding_block,
            add_winding_carry,
            mark_tail_winding_offsets,
            scan_offset_block,
            add_offset_carry,
            emit_cell_entries,
            update_metadata,
        }
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

struct BindGroups {
    split_quadcell: [wgpu::BindGroup; 2],
    split_cell_entry: [wgpu::BindGroup; 2],
    mark_tail: wgpu::BindGroup,
    offset_scan_bgs: Vec<wgpu::BindGroup>,
    emit_result: wgpu::BindGroup,
    winding_scan_bgs: Vec<wgpu::BindGroup>,
    update_metadata: [wgpu::BindGroup; 2],
}

impl BindGroups {
    fn new(device: &wgpu::Device, resources: &Resources, pipelines: &Pipelines) -> Self {
        let Resources {
            cell_metadata_buffer_1,
            cell_metadata_buffer_2,
            cell_entries_buffer,
            segments_buffer,
            // intermediates
            split_entries_buffer,
            cell_offsets_buffer,
            winding_block_sum_buffers,
            winding_scan_params_buffers,
            offset_block_sum_buffers,
            offset_scan_params_buffers,
            // result info
            result_info_buffer,
            ..
        } = resources;

        let Pipelines {
            quadcell_split,
            build_split_entries: build_split,
            scan_winding_block,
            mark_tail_winding_offsets,
            scan_offset_block,
            emit_cell_entries,
            update_metadata,
            ..
        } = pipelines;

        // QuadCell split: cell_metadata ping-pong (reads parent metadata, writes child metadata).
        // binding(2) = result_info for min_seg threshold used in split skipping.
        let split_quadcell_ping = device.create_bind_group(&BindGroupDescriptor {
            label: Some("split quadcell ping bind group"),
            layout: &quadcell_split.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_metadata_buffer_1),
                bg_entry(1, cell_metadata_buffer_2),
                bg_entry(2, result_info_buffer),
            ],
        });
        let split_quadcell_pong = device.create_bind_group(&BindGroupDescriptor {
            label: Some("split quadcell pong bind group"),
            layout: &quadcell_split.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_metadata_buffer_2),
                bg_entry(1, cell_metadata_buffer_1),
                bg_entry(2, result_info_buffer),
            ],
        });

        // Initial Split: build_split_entries reads cell_entries + cell_metadata (ping-pong for metadata).
        let split_cell_entry_ping = device.create_bind_group(&BindGroupDescriptor {
            label: Some("split cell entry ping bind group"),
            layout: &build_split.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_entries_buffer),
                bg_entry(1, segments_buffer),
                bg_entry(2, cell_metadata_buffer_1),
                bg_entry(3, split_entries_buffer),
                bg_entry(4, cell_offsets_buffer),
                bg_entry(5, &winding_block_sum_buffers[0]),
                bg_entry(6, result_info_buffer),
            ],
        });
        let split_cell_entry_pong = device.create_bind_group(&BindGroupDescriptor {
            label: Some("split cell entry pong bind group"),
            layout: &build_split.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_entries_buffer),
                bg_entry(1, segments_buffer),
                bg_entry(2, cell_metadata_buffer_2),
                bg_entry(3, split_entries_buffer),
                bg_entry(4, cell_offsets_buffer),
                bg_entry(5, &winding_block_sum_buffers[0]),
                bg_entry(6, result_info_buffer),
            ],
        });

        let mut winding_scan_bgs = Vec::new();
        for i in 0..winding_block_sum_buffers.len() - 1 {
            winding_scan_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: Some("winding scan bind group"),
                layout: &scan_winding_block.get_bind_group_layout(0),
                entries: &[
                    bg_entry(0, cell_entries_buffer),
                    bg_entry(1, split_entries_buffer),
                    bg_entry(2, cell_offsets_buffer),
                    bg_entry(3, &winding_block_sum_buffers[i]),
                    bg_entry(4, &winding_block_sum_buffers[i + 1]),
                    bg_entry(5, result_info_buffer),
                    bg_entry(6, &winding_scan_params_buffers[i]),
                ],
            }));
        }

        let mark_tail = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mark tail bind group"),
            layout: &mark_tail_winding_offsets.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_entries_buffer),
                bg_entry(1, split_entries_buffer),
                bg_entry(2, cell_offsets_buffer),
                bg_entry(3, &winding_block_sum_buffers[0]),
                bg_entry(4, &winding_block_sum_buffers[1]),
                bg_entry(5, result_info_buffer),
                bg_entry(6, &winding_scan_params_buffers[0]),
            ],
        });

        let mut offset_scan_bgs: Vec<BindGroup> = vec![];
        let mut offset_levels: Vec<&Buffer> = vec![cell_offsets_buffer];
        offset_levels.extend(offset_block_sum_buffers.iter());
        for i in 0..offset_levels.len() - 1 {
            offset_scan_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: Some("offsets scan bind group"),
                layout: &scan_offset_block.get_bind_group_layout(0),
                entries: &[
                    bg_entry(0, offset_levels[i]),
                    bg_entry(1, offset_levels[i + 1]),
                    bg_entry(2, &offset_scan_params_buffers[i]),
                ],
            }));
        }

        // Emit result: emit_cell_entries writes to cell_entries (in-place overwrite)
        let emit_result = device.create_bind_group(&BindGroupDescriptor {
            label: Some("emit result bind group"),
            layout: &emit_cell_entries.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_entries_buffer),
                bg_entry(1, split_entries_buffer),
                bg_entry(2, cell_offsets_buffer),
                bg_entry(3, result_info_buffer),
            ],
        });

        // Update metadata: reads cell_entries, writes to output cell_metadata (ping-pong for metadata)
        let update_metadata_ping = device.create_bind_group(&BindGroupDescriptor {
            label: Some("update metadata ping bind group"),
            layout: &update_metadata.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_entries_buffer),
                bg_entry(1, cell_metadata_buffer_2), // even depth: metadata output is buffer_2
                bg_entry(2, result_info_buffer),
            ],
        });
        let update_metadata_pong = device.create_bind_group(&BindGroupDescriptor {
            label: Some("update metadata pong bind group"),
            layout: &update_metadata.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, cell_entries_buffer),
                bg_entry(1, cell_metadata_buffer_1), // odd depth: metadata output is buffer_1
                bg_entry(2, result_info_buffer),
            ],
        });

        Self {
            split_quadcell: [split_quadcell_ping, split_quadcell_pong],
            split_cell_entry: [split_cell_entry_ping, split_cell_entry_pong],
            mark_tail,
            winding_scan_bgs,
            offset_scan_bgs,
            emit_result,
            update_metadata: [update_metadata_ping, update_metadata_pong],
        }
    }
}

fn dispatch_for_items(items: u32, max_dim: u32) -> [u32; 3] {
    let wg = items.max(1).div_ceil(WG_SIZE);
    split_dispatch_3d(wg, max_dim)
}

/// Compute the number of elements at each hierarchical scan level.
/// Starting from `initial` elements, each level reduces by WG_SIZE.
fn hierarchical_level_counts(initial: u32, levels: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(levels);
    let mut n = initial;
    for _ in 0..levels {
        out.push(n);
        if n <= 1 {
            break;
        }
        n = n.div_ceil(WG_SIZE);
    }
    out
}

pub struct QuadTreeGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    resources: Resources,
    bind_groups: BindGroups,
    num_cell_entries: u32,
    // Minimum entry count for a cell to be split further (passed to quadcell_split.wgsl).
    min_seg: u32,
}

impl QuadTreeGpuContext {
    pub async fn new(
        cell_entries: &[CellEntry],
        segments: &[AbstractLineSegment],
        parent_bound: &Rect,
        max_depth: u8,
        min_seg: u32,
    ) -> anyhow::Result<Self> {
        let (device, queue) = init_wgpu().await;

        let pipelines = Pipelines::new(&device);
        let resources = Resources::new(&device, &cell_entries, &segments, max_depth);
        let bind_groups = BindGroups::new(&device, &resources, &pipelines);
        // Write initial data
        let root_meta = CellMetadata::new(parent_bound, 0, cell_entries.len() as u32);
        queue.write_buffer(
            &resources.cell_metadata_buffer_1,
            0,
            bytemuck::cast_slice(&[root_meta]),
        );
        queue.write_buffer(
            &resources.cell_entries_buffer,
            0,
            bytemuck::cast_slice(cell_entries),
        );
        Ok(Self {
            device,
            queue,
            pipelines,
            resources,
            bind_groups,
            num_cell_entries: cell_entries.len() as u32,
            min_seg,
        })
    }

    /// Process one level of quad-tree subdivision.
    /// 1. Split quad cells (compute child bboxes and write to cell_metadata_out)
    /// 2. Run entry subdivision kernels 1-5 (split entries among child cells)
    ///
    /// `num_entries` is the actual number of entries in `cell_entries_buffer` for this depth.
    /// It is written into `result_info` before any shader runs so that shaders can read it
    /// instead of relying on `arrayLength()`.
    pub fn process_level(&self, depth: u8, num_cells: u32, num_entries: u32) {
        let max_dim = self.device.limits().max_compute_workgroups_per_dimension;
        let ping = (depth % 2) as usize;
        let num_offsets = num_entries.saturating_mul(4);
        let max_result_entries = num_offsets; // each entry can split into at most 4 child entries
        let winding_levels =
            hierarchical_level_counts(num_entries, self.bind_groups.winding_scan_bgs.len());
        let offset_levels =
            hierarchical_level_counts(num_offsets, self.bind_groups.offset_scan_bgs.len());

        // Write the actual entry count and min_seg threshold into result_info.
        // This must happen before the compute encoder so the write is visible to all kernels.
        self.queue.write_buffer(
            &self.resources.result_info_buffer,
            0,
            bytemuck::cast_slice(&[SplitResultInfo {
                cell_entries_length: num_entries,
                min_seg: self.min_seg,
                _pad: [0; 2],
            }]),
        );
        for (i, &level_len) in winding_levels.iter().enumerate() {
            self.queue.write_buffer(
                &self.resources.winding_scan_params_buffers[i],
                0,
                bytes_of(&ScanParams {
                    level_len,
                    carry_len: level_len.div_ceil(WG_SIZE),
                    _pad: [0; 2],
                }),
            );
        }
        for (i, &level_len) in offset_levels.iter().enumerate() {
            self.queue.write_buffer(
                &self.resources.offset_scan_params_buffers[i],
                0,
                bytes_of(&ScanParams {
                    level_len,
                    carry_len: level_len.div_ceil(WG_SIZE),
                    _pad: [0; 2],
                }),
            );
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Clear intermediate buffers from the previous level to avoid stale data.
        // cell_offsets: 4 interleaved offset arrays, all must start at 0 before scanning.
        encoder.clear_buffer(&self.resources.cell_offsets_buffer, 0, None);
        // winding_block_sum level-0: initial per-entry winding accumulators must start clean.
        encoder.clear_buffer(&self.resources.winding_block_sum_buffers[0], 0, None);

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());

            // QuadCell split
            pass.set_pipeline(&self.pipelines.quadcell_split);
            pass.set_bind_group(0, &self.bind_groups.split_quadcell[ping], &[]);
            let [x, y, z] = split_dispatch_3d(num_cells, max_dim);
            pass.dispatch_workgroups(x, y, z);

            // Build split entries
            pass.set_pipeline(&self.pipelines.build_split_entries);
            pass.set_bind_group(0, &self.bind_groups.split_cell_entry[ping], &[]);
            let [x, y, z] = dispatch_for_items(num_entries, max_dim);
            pass.dispatch_workgroups(x, y, z);

            // Winding hierarchy forward scan + backward carry
            let winding_bgs = &self.bind_groups.winding_scan_bgs;
            for i in 0..winding_levels.len() {
                pass.set_pipeline(&self.pipelines.scan_winding_block);
                pass.set_bind_group(0, &winding_bgs[i], &[]);
                let [x, y, z] = dispatch_for_items(winding_levels[i], max_dim);
                pass.dispatch_workgroups(x, y, z);
            }
            for i in (0..winding_levels.len().saturating_sub(1)).rev() {
                pass.set_pipeline(&self.pipelines.add_winding_carry);
                pass.set_bind_group(0, &winding_bgs[i], &[]);
                let [x, y, z] = dispatch_for_items(winding_levels[i], max_dim);
                pass.dispatch_workgroups(x, y, z);
            }

            // Mark tail winding offsets
            pass.set_pipeline(&self.pipelines.mark_tail_winding_offsets);
            pass.set_bind_group(0, &self.bind_groups.mark_tail, &[]);
            let [x, y, z] = dispatch_for_items(num_entries, max_dim);
            pass.dispatch_workgroups(x, y, z);

            // Offset hierarchy forward scan + backward carry
            let offset_bgs = &self.bind_groups.offset_scan_bgs;
            for i in 0..offset_levels.len() {
                pass.set_pipeline(&self.pipelines.scan_offset_block);
                pass.set_bind_group(0, &offset_bgs[i], &[]);
                let [x, y, z] = dispatch_for_items(offset_levels[i], max_dim);
                pass.dispatch_workgroups(x, y, z);
            }
            for i in (0..offset_levels.len().saturating_sub(1)).rev() {
                pass.set_pipeline(&self.pipelines.add_offset_carry);
                pass.set_bind_group(0, &offset_bgs[i], &[]);
                let [x, y, z] = dispatch_for_items(offset_levels[i], max_dim);
                pass.dispatch_workgroups(x, y, z);
            }

            // Emit cell entries (writes to cell_entries buffer in-place)
            pass.set_pipeline(&self.pipelines.emit_cell_entries);
            pass.set_bind_group(0, &self.bind_groups.emit_result, &[]);
            let [x, y, z] = dispatch_for_items(num_offsets, max_dim);
            pass.dispatch_workgroups(x, y, z);

            // Update metadata: write entry_start/entry_count into cell_metadata_out.
            // Actual result count is only known on GPU (result_info), so dispatch by
            // max_result_entries as upper bound; shader early-returns for out-of-range threads.
            pass.set_pipeline(&self.pipelines.update_metadata);
            pass.set_bind_group(0, &self.bind_groups.update_metadata[ping], &[]);
            let [x, y, z] = split_dispatch_3d(max_result_entries.max(1), max_dim);
            pass.dispatch_workgroups(x, y, z);
        }
        self.queue.submit([encoder.finish()]);
    }

    pub fn readback<T: AnyBitPattern>(
        &self,
        source_buffer: &wgpu::Buffer,
        readback_buffer: &wgpu::Buffer,
    ) -> anyhow::Result<Vec<T>> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&source_buffer, 0, readback_buffer, 0, source_buffer.size());
        self.queue.submit([encoder.finish()]);
        let slice = readback_buffer.slice(..);
        let (tx, rx) = channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;

        let bytes = slice.get_mapped_range();
        let out_u32: &[T] = bytemuck::cast_slice(&bytes);
        let v = out_u32.to_vec();
        drop(bytes);
        readback_buffer.unmap();
        Ok(v)
    }

    pub fn print_offsets(&self) -> anyhow::Result<()> {
        let offsets = self.readback::<u32>(
            &self.resources.cell_offsets_buffer,
            &self.resources.cell_offsets_readback_buffer,
        )?;
        println!("=== GPU: CellEntry offsets ===");
        println!("{:?}", offsets);
        Ok(())
    }

    pub fn read_cell_entry(&self) -> anyhow::Result<Vec<CellEntry>> {
        self.readback::<CellEntry>(
            &self.resources.cell_entries_buffer,
            &self.resources.cell_entry_readback_buffer,
        )
    }

    pub fn read_result_info(&self) -> anyhow::Result<SplitResultInfo> {
        let res = self.readback::<SplitResultInfo>(
            &self.resources.result_info_buffer,
            &self.resources.result_info_readback_buffer,
        )?;
        Ok(res[0])
    }

    /// Read cell metadata from the output buffer after all levels have been processed.
    /// The last process_level call uses depth = max_depth - 1, whose output goes to:
    ///   even last_depth -> buffer_2, odd last_depth -> buffer_1
    pub fn read_cell_metadata(&self, last_depth: u8) -> anyhow::Result<Vec<CellMetadata>> {
        let source_buffer = if last_depth % 2 == 0 {
            &self.resources.cell_metadata_buffer_2
        } else {
            &self.resources.cell_metadata_buffer_1
        };
        self.readback::<CellMetadata>(source_buffer, &self.resources.cell_metadata_readback_buffer)
    }

    pub fn read_winding_block_sums(&self) -> anyhow::Result<Vec<Vec<WindingBlockInfo>>> {
        self.resources
            .winding_block_sum_buffers
            .iter()
            .zip(self.resources.winding_block_sum_readback_buffers.iter())
            .map(|(src, dst)| self.readback::<WindingBlockInfo>(src, dst))
            .collect()
    }

    pub fn print_winding_block_sums(&self) -> anyhow::Result<()> {
        let levels = self.read_winding_block_sums()?;
        for (level, infos) in levels.iter().enumerate() {
            println!("=== GPU: Winding Block Sums Level {level} ===");
            for (idx, info) in infos.iter().enumerate() {
                println!("[{idx}] {:?}", info);
            }
        }
        Ok(())
    }

    pub fn print_split_entries(&self) -> anyhow::Result<()> {
        let entries = self.readback::<SplitEntry>(
            &self.resources.split_entries_buffer,
            &self.resources.split_entries_readback_buffer,
        )?;
        println!("=== GPU: Split entries ===");
        print_split_entries(&entries);
        Ok(())
    }
}
