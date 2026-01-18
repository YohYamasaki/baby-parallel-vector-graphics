use crate::abstract_segment::AbstractLineSegment;
use crate::cell_entry::{print_split_entries, CellEntry, SplitEntry};
use crate::geometry::rect::Rect;
use crate::gpu::init::init_wgpu;
use bytemuck::{bytes_of, AnyBitPattern, Pod, Zeroable};
use std::slice;
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
pub struct ParentCellBound {
    bbox_ltrb: [f32; 4],
    mid_x: f32,
    mid_y: f32,
    _pad: [u32; 2],
}

impl ParentCellBound {
    pub fn new(rect: &Rect) -> Self {
        let [mid_x, mid_y] = rect.mid_point();
        Self {
            bbox_ltrb: [rect.left(), rect.top(), rect.right(), rect.bottom()],
            mid_x,
            mid_y,
            _pad: [0u32; 2],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct SplitResultInfo {
    pub cell_entries_length: u32,
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
    // inputs
    input_cell_entries_buffer: wgpu::Buffer,
    segments_buffer: wgpu::Buffer,
    parent_bounds_buffer: wgpu::Buffer,
    // intermediates
    split_entries_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer,
    winding_block_sum_buffers: Vec<Buffer>,
    offset_block_sum_buffers: Vec<Buffer>,
    // results
    result_cell_entries_buffer: wgpu::Buffer,
    result_info_buffer: wgpu::Buffer,
    // readbacks
    winding_block_sum_readback_buffers: Vec<Buffer>,
    split_entries_readback_buffer: wgpu::Buffer,
    cell_offsets_readback_buffer: wgpu::Buffer,
    result_entries_readback_buffer: wgpu::Buffer,
    result_info_readback_buffer: wgpu::Buffer,
}

impl Resources {
    fn new(
        device: &wgpu::Device,
        parent_bound: &ParentCellBound,
        cell_entries: &[CellEntry],
        segments: &[AbstractLineSegment],
    ) -> Self {
        // inputs
        let input_cell_entries_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("input cell entries buffer"),
            contents: bytemuck::cast_slice(cell_entries),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let parent_bounds_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("parent bound buffer"),
            contents: bytemuck::bytes_of(parent_bound),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
        });
        let segments_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("segments buffer"),
            contents: bytemuck::cast_slice(segments),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        // intermediate structures
        let split_entries_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("split entries buffer"),
            size: (cell_entries.len() * size_of::<SplitEntry>()) as u64, // TODO: enough length?
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_offsets_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("parent bounds buffer"),
            size: ((cell_entries.len() * 4 * size_of::<u32>()) as u64).max(size_of::<u32>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create buffers for each level since the prefix sum process is done recursively the following process.
        // Scan by blocks using Hillis-Steele -> add carry from the one previous block's last element
        let create_sum_buffer = |bytes: u64| {
            device.create_buffer(&BufferDescriptor {
                label: Some("winding block sum buffer"),
                size: bytes.max(32),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cell_entries_bytes = (size_of::<WindingBlockInfo>() * cell_entries.len()) as u64;
        let mut winding_block_sum_buffers: Vec<Buffer> =
            vec![create_sum_buffer(cell_entries_bytes)];
        let mut level_elms = cell_entries.len();
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
            device.create_buffer(&BufferDescriptor {
                label: Some("offset block sum buffer"),
                size: bytes.max(size_of::<u32>() as u64),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mut offset_block_sum_buffers: Vec<Buffer> = vec![];
        let mut offset_level_elms = cell_entries.len().saturating_mul(4).max(1);
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

        // results
        let result_cell_entries_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("output cell entries buffer"),
            size: (cell_entries.len() * 4 * size_of::<CellEntry>()) as u64, // input cell entries can be split into *4 number of child cells
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_info_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("result info buffer"),
            size: size_of::<SplitResultInfo>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // readback buffers
        let result_entries_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug out buffer"),
            size: result_cell_entries_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let split_entries_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("parent bounds buffer"),
            size: split_entries_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let cell_offsets_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug out buffer"),
            size: cell_offsets_buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let result_info_readback_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("result info readback buffer"),
            size: result_info_buffer.size(),
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
            // inputs
            input_cell_entries_buffer,
            segments_buffer,
            parent_bounds_buffer,
            // intermediates
            split_entries_buffer,
            cell_offsets_buffer,
            winding_block_sum_buffers,
            offset_block_sum_buffers,
            // results
            result_info_buffer,
            result_cell_entries_buffer,
            // readbacks
            winding_block_sum_readback_buffers,
            cell_offsets_readback_buffer,
            split_entries_readback_buffer,
            result_info_readback_buffer,
            result_entries_readback_buffer,
        }
    }
}

struct Pipelines {
    build_split: wgpu::ComputePipeline,
    scan_winding_block: wgpu::ComputePipeline,
    scan_offset_block: wgpu::ComputePipeline,
    add_offset_carry: wgpu::ComputePipeline,
    emit_cell_entries: wgpu::ComputePipeline,
    mark_tail_winding_offsets: wgpu::ComputePipeline,
    add_winding_carry: wgpu::ComputePipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let split_shader = device.create_shader_module(ShaderModuleDescriptor {
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

        let winding_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("winding bind group"),
            entries: &[
                bgl_storage_entry(0),
                bgl_storage_entry(1),
                bgl_storage_entry(2),
                bgl_storage_entry(3),
                bgl_storage_entry(4),
            ],
        });

        let winding_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("winding pl"),
            bind_group_layouts: &[&winding_bgl],
            immediate_size: 0,
        });

        let offset_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("offset bind group"),
            entries: &[bgl_storage_entry(0), bgl_storage_entry(1)],
        });
        let offset_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("offset pl"),
            bind_group_layouts: &[&offset_bgl],
            immediate_size: 0,
        });

        let build_split = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("split pipeline"),
            layout: None,
            module: &split_shader,
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

        Self {
            build_split,
            scan_winding_block,
            add_winding_carry,
            mark_tail_winding_offsets,
            scan_offset_block,
            add_offset_carry,
            emit_cell_entries,
        }
    }

    fn get(&self, id: PipelineId) -> &wgpu::ComputePipeline {
        match id {
            PipelineId::BuildSplit => &self.build_split,
            PipelineId::ScanWindingBlock => &self.scan_winding_block,
            PipelineId::AddWindingCarry => &self.add_winding_carry,
            PipelineId::MarkTailWindingOffsets => &self.mark_tail_winding_offsets,
            PipelineId::ScanOffsetBlock => &self.scan_offset_block,
            PipelineId::AddOffsetCarry => &self.add_offset_carry,
            PipelineId::EmitCellEntries => &self.emit_cell_entries,
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
    split: wgpu::BindGroup,
    mark_tail: wgpu::BindGroup,
    offset_scan_bgs: Vec<wgpu::BindGroup>,
    emit_result: wgpu::BindGroup,
    winding_scan_bgs: Vec<wgpu::BindGroup>,
}

impl BindGroups {
    fn new(device: &wgpu::Device, resources: &Resources, pipelines: &Pipelines) -> Self {
        let Resources {
            // inputs
            input_cell_entries_buffer,
            segments_buffer,
            parent_bounds_buffer,
            // intermediates
            split_entries_buffer,
            cell_offsets_buffer,
            winding_block_sum_buffers,
            offset_block_sum_buffers,
            // results
            result_cell_entries_buffer,
            result_info_buffer,
            ..
        } = resources;

        let Pipelines {
            build_split,
            scan_winding_block,
            mark_tail_winding_offsets,
            scan_offset_block,
            emit_cell_entries,
            ..
        } = pipelines;

        let split = device.create_bind_group(&BindGroupDescriptor {
            label: Some("split bind group"),
            layout: &build_split.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, input_cell_entries_buffer),
                bg_entry(1, segments_buffer),
                bg_entry(2, parent_bounds_buffer),
                bg_entry(3, split_entries_buffer),
                bg_entry(4, cell_offsets_buffer),
                bg_entry(5, &winding_block_sum_buffers[0]),
            ],
        });
        let mut winding_scan_bgs: Vec<BindGroup> = vec![];
        for i in 0..winding_block_sum_buffers.len() - 1 {
            winding_scan_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: Some("winding scan bind group"),
                layout: &scan_winding_block.get_bind_group_layout(0),
                entries: &[
                    bg_entry(0, input_cell_entries_buffer),
                    bg_entry(1, split_entries_buffer),
                    bg_entry(2, cell_offsets_buffer),
                    bg_entry(3, &winding_block_sum_buffers[i]),
                    bg_entry(4, &winding_block_sum_buffers[i + 1]),
                ],
            }));
        }
        let mark_tail = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mark tail bind group"),
            layout: &mark_tail_winding_offsets.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, input_cell_entries_buffer),
                bg_entry(1, split_entries_buffer),
                bg_entry(2, cell_offsets_buffer),
                bg_entry(3, &winding_block_sum_buffers[0]),
                bg_entry(4, &winding_block_sum_buffers[1]),
            ],
        });
        let mut offset_scan_bgs: Vec<BindGroup> = vec![];
        let mut offset_levels: Vec<&Buffer> = vec![cell_offsets_buffer];
        offset_levels.extend(offset_block_sum_buffers.iter());
        for i in 0..offset_levels.len() - 1 {
            offset_scan_bgs.push(device.create_bind_group(&BindGroupDescriptor {
                label: Some("offsets scan bind group"),
                layout: &scan_offset_block.get_bind_group_layout(0),
                entries: &[bg_entry(0, offset_levels[i]), bg_entry(1, offset_levels[i + 1])],
            }));
        }
        let emit_result = device.create_bind_group(&BindGroupDescriptor {
            label: Some("split to cell bind group"),
            layout: &emit_cell_entries.get_bind_group_layout(0),
            entries: &[
                bg_entry(0, result_cell_entries_buffer),
                bg_entry(1, split_entries_buffer),
                bg_entry(2, cell_offsets_buffer),
                bg_entry(3, result_info_buffer),
            ],
        });

        Self {
            split,
            mark_tail,
            winding_scan_bgs,
            offset_scan_bgs,
            emit_result,
        }
    }

    fn get(&self, id: BindGroupId) -> &[wgpu::BindGroup] {
        match id {
            BindGroupId::Split => slice::from_ref(&self.split),
            BindGroupId::MarkTail => slice::from_ref(&self.mark_tail),
            BindGroupId::EmitResult => slice::from_ref(&self.emit_result),
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum PipelineId {
    BuildSplit,
    ScanWindingBlock,
    ScanOffsetBlock,
    AddOffsetCarry,
    EmitCellEntries,
    AddWindingCarry,
    MarkTailWindingOffsets,
}

#[derive(Debug, Copy, Clone)]
enum BindGroupId {
    Split,
    MarkTail,
    EmitResult,
    // BlockSums,
}

#[derive(Debug, Copy, Clone)]
enum DispatchKind {
    ByEntries, // num_entries
    ByOffsets, // num_entries * 4
}

#[derive(Debug, Copy, Clone)]
enum PassSpec {
    Single {
        pipeline: PipelineId,
        bind_group: BindGroupId,
        dispatch: DispatchKind,
    },
    WindingHerarchy,
    OffsetHerarchy,
}

#[derive(Debug, Copy, Clone)]
struct RunMeta {
    num_entries: u32,
    num_offsets: u32,
}

impl RunMeta {
    fn new(num_entries: u32) -> Self {
        let num_offsets = num_entries.saturating_mul(4);
        Self {
            num_entries,
            num_offsets,
        }
    }

    fn item_count(self, kind: DispatchKind) -> u32 {
        match kind {
            DispatchKind::ByEntries => self.num_entries.max(1),
            DispatchKind::ByOffsets => self.num_offsets.max(1),
        }
    }

    fn winding_level_entries(self, levels: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(levels);
        let mut n = self.num_entries.max(1);
        for _ in 0..levels {
            out.push(n);
            n = n.div_ceil(WG_SIZE).max(1);
        }
        out
    }

    fn offset_level_entries(self, levels: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(levels);
        let mut n = self.num_offsets.max(1);
        for _ in 0..levels {
            out.push(n);
            n = n.div_ceil(WG_SIZE).max(1);
        }
        out
    }
}

const PASS_GRAPH: &[PassSpec] = &[
    PassSpec::Single {
        pipeline: PipelineId::BuildSplit,
        bind_group: BindGroupId::Split,
        dispatch: DispatchKind::ByEntries,
    },
    PassSpec::WindingHerarchy,
    PassSpec::Single {
        pipeline: PipelineId::MarkTailWindingOffsets,
        bind_group: BindGroupId::MarkTail,
        dispatch: DispatchKind::ByEntries,
    },
    PassSpec::OffsetHerarchy,
    PassSpec::Single {
        pipeline: PipelineId::EmitCellEntries,
        bind_group: BindGroupId::EmitResult,
        dispatch: DispatchKind::ByOffsets,
    },
];

fn dispatch_for_items(items: u32, max_dim: u32) -> [u32; 3] {
    let wg = items.max(1).div_ceil(WG_SIZE);
    split_dispatch_3d(wg, max_dim)
}

fn dispatch_for(kind: DispatchKind, meta: RunMeta, max_dim: u32) -> [u32; 3] {
    dispatch_for_items(meta.item_count(kind), max_dim)
}

pub struct CellEntryGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    resources: Resources,
    bind_groups: BindGroups,
    num_cell_entries: u32,
}

impl CellEntryGpuContext {
    pub async fn new(
        cell_entries: &[CellEntry],
        segments: &[AbstractLineSegment],
        parent_bound: &ParentCellBound,
    ) -> anyhow::Result<Self> {
        let (device, queue) = init_wgpu().await;

        let pipelines = Pipelines::new(&device);
        let resources = Resources::new(&device, &parent_bound, &cell_entries, &segments);
        let bind_groups = BindGroups::new(&device, &resources, &pipelines);

        Ok(Self {
            device,
            queue,
            pipelines,
            resources,
            bind_groups,
            num_cell_entries: cell_entries.len() as u32,
        })
    }

    fn run_winding_hierarchy(&self, pass: &mut wgpu::ComputePass<'_>, meta: RunMeta, max_dim: u32) {
        let bgs = &self.bind_groups.winding_scan_bgs;
        let level_entries = meta.winding_level_entries(bgs.len());

        // forward: scan + block_sum
        for i in 0..bgs.len() {
            pass.set_pipeline(self.pipelines.get(PipelineId::ScanWindingBlock));
            pass.set_bind_group(0, &bgs[i], &[]);
            let [x, y, z] = dispatch_for_items(level_entries[i], max_dim);
            pass.dispatch_workgroups(x, y, z);
        }

        // reverse: add carry (top level is parentなしなので除外)
        for i in (0..bgs.len().saturating_sub(1)).rev() {
            pass.set_pipeline(self.pipelines.get(PipelineId::AddWindingCarry));
            pass.set_bind_group(0, &bgs[i], &[]);
            let [x, y, z] = dispatch_for_items(level_entries[i], max_dim);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    fn run_offset_hierarchy(&self, pass: &mut wgpu::ComputePass<'_>, meta: RunMeta, max_dim: u32) {
        let bgs = &self.bind_groups.offset_scan_bgs;
        let level_entries = meta.offset_level_entries(bgs.len());

        // forward: scan + block_sum
        for i in 0..bgs.len() {
            pass.set_pipeline(self.pipelines.get(PipelineId::ScanOffsetBlock));
            pass.set_bind_group(0, &bgs[i], &[]);
            let [x, y, z] = dispatch_for_items(level_entries[i], max_dim);
            pass.dispatch_workgroups(x, y, z);
        }

        // reverse: add carry (top level has no parent carry)
        for i in (0..bgs.len().saturating_sub(1)).rev() {
            pass.set_pipeline(self.pipelines.get(PipelineId::AddOffsetCarry));
            pass.set_bind_group(0, &bgs[i], &[]);
            let [x, y, z] = dispatch_for_items(level_entries[i], max_dim);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    pub fn run_subdivision(&self) {
        let max_dim = self.device.limits().max_compute_workgroups_per_dimension;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let meta = RunMeta::new(self.num_cell_entries);
            let mut pass = encoder.begin_compute_pass(&Default::default());

            for spec in PASS_GRAPH {
                match spec {
                    PassSpec::Single {
                        pipeline,
                        bind_group,
                        dispatch,
                        ..
                    } => {
                        pass.set_pipeline(self.pipelines.get(*pipeline));
                        self.bind_groups.get(*bind_group).iter().for_each(|bg| {
                            pass.set_bind_group(0, bg, &[]);
                            let [x, y, z] = dispatch_for(*dispatch, meta, max_dim);
                            pass.dispatch_workgroups(x, y, z);
                        });
                    }
                    PassSpec::WindingHerarchy => {
                        self.run_winding_hierarchy(&mut pass, meta, max_dim);
                    }
                    PassSpec::OffsetHerarchy => {
                        self.run_offset_hierarchy(&mut pass, meta, max_dim);
                    }
                }
            }
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
            &self.resources.result_cell_entries_buffer,
            &self.resources.result_entries_readback_buffer,
        )
    }

    pub fn read_result_info(&self) -> anyhow::Result<SplitResultInfo> {
        let res = self.readback::<SplitResultInfo>(
            &self.resources.result_info_buffer,
            &self.resources.result_info_readback_buffer,
        )?;
        Ok(res[0])
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
