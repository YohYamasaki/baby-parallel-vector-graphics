use crate::abstract_segment::AbstractLineSegment;
use crate::cell_entry::CellEntry;
use crate::gpu::quad_tree::CellMetadata;
use crate::path::{AbstractPath, Paint};
use anyhow::Context;
use bytemuck::{bytes_of, Pod, Zeroable};
use std::sync::mpsc::channel;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Extent3d, Features, MapMode, PipelineCompilationOptions, PollType,
    PowerPreference, Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, Surface,
    SurfaceConfiguration, SurfaceError, SurfaceTexture, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

const RENDER_WG_SIZE_X: u32 = 8;
const RENDER_WG_SIZE_Y: u32 = 8;

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct PathPaintGpu {
    rgba: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct RenderParams {
    width: u32,
    height: u32,
    entries_len: u32,
    _pad: u32,
}

pub fn build_path_paints(abs_paths: &[AbstractPath], paints: &[Paint]) -> Vec<PathPaintGpu> {
    let mut out = Vec::with_capacity(abs_paths.len().max(1));
    for path in abs_paths {
        let rgba = paints
            .get(path.paint_id)
            .map(|paint| match paint {
                Paint::SolidColor { rgba } => *rgba,
            })
            .unwrap_or([0, 0, 0, 255]);
        out.push(PathPaintGpu {
            rgba: [
                rgba[0] as f32 / 255.0,
                rgba[1] as f32 / 255.0,
                rgba[2] as f32 / 255.0,
                rgba[3] as f32 / 255.0,
            ],
        });
    }
    if out.is_empty() {
        out.push(PathPaintGpu {
            rgba: [0.0, 0.0, 0.0, 1.0],
        });
    }
    out
}

pub struct ComputeRenderer {
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pipeline: ComputePipeline,
    output_texture: Texture,
    output_view: TextureView,
    blitter: wgpu::util::TextureBlitter,
}

impl ComputeRenderer {
    pub async fn new(
        instance: &wgpu::Instance,
        surface: &Surface<'_>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
            .context("No surface-compatible adapter found")?;

        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("gpu renderer device"),
                required_features: Features::empty(),
                required_limits: limits,
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .context("Failed to create renderer device")?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let present_mode = caps
            .present_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::PresentMode::Fifo)
            .unwrap_or(caps.present_modes[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("cell render compute shader"),
            source: ShaderSource::Wgsl(include_str!("cell_render.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("cell render pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let (output_texture, output_view) =
            create_output_texture(&device, config.width, config.height);
        let blitter = wgpu::util::TextureBlitter::new(&device, config.format);

        Ok(Self {
            device,
            queue,
            config,
            pipeline,
            output_texture,
            output_view,
            blitter,
        })
    }

    pub fn render_to_rgba(
        &self,
        surface: &Surface<'_>,
        cell_metadata: &[CellMetadata],
        cell_entries: &[CellEntry],
        segments: &[AbstractLineSegment],
        path_paints: &[PathPaintGpu],
    ) -> anyhow::Result<Vec<u8>> {
        let metadata_buffer =
            create_storage_buffer_or_dummy(&self.device, "renderer metadata buffer", cell_metadata);
        let entries_buffer = create_storage_buffer_or_dummy(
            &self.device,
            "renderer cell entries buffer",
            cell_entries,
        );
        let segments_buffer =
            create_storage_buffer_or_dummy(&self.device, "renderer segments buffer", segments);
        let path_paints_buffer = create_storage_buffer_or_dummy(
            &self.device,
            "renderer path paints buffer",
            path_paints,
        );

        let params = RenderParams {
            width: self.config.width,
            height: self.config.height,
            entries_len: cell_entries.len() as u32,
            _pad: 0,
        };
        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("renderer params buffer"),
            contents: bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bg = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("cell render bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: metadata_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: entries_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: segments_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: path_paints_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&self.output_view),
                },
            ],
        });

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = self.config.width * bytes_per_pixel;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output_size = (padded_bytes_per_row * self.config.height) as u64;
        let readback_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("renderer readback buffer"),
            size: output_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("renderer command encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cell render pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let x = self.config.width.div_ceil(RENDER_WG_SIZE_X);
            let y = self.config.height.div_ceil(RENDER_WG_SIZE_Y);
            pass.dispatch_workgroups(x, y, 1);
        }

        let mut frame_to_present: Option<SurfaceTexture> = None;
        match surface.get_current_texture() {
            Ok(frame) => {
                {
                    let view = frame.texture.create_view(&TextureViewDescriptor::default());
                    self.blitter
                        .copy(&self.device, &mut encoder, &self.output_view, &view);
                }
                frame_to_present = Some(frame);
            }
            Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                surface.configure(&self.device, &self.config);
            }
            Err(SurfaceError::Timeout) => {}
            Err(SurfaceError::OutOfMemory) => {
                anyhow::bail!("surface out of memory");
            }
            Err(SurfaceError::Other) => {}
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.config.height),
                },
            },
            Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        if let Some(frame) = frame_to_present {
            frame.present();
        }

        let slice = readback_buffer.slice(..);
        let (tx, rx) = channel();
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        self.device.poll(PollType::wait_indefinitely())?;
        rx.recv()??;

        let data = slice.get_mapped_range();
        let mut rgba =
            vec![0u8; (self.config.width * self.config.height * bytes_per_pixel) as usize];
        for row in 0..self.config.height as usize {
            let src_offset = row * padded_bytes_per_row as usize;
            let dst_offset = row * unpadded_bytes_per_row as usize;
            rgba[dst_offset..dst_offset + unpadded_bytes_per_row as usize]
                .copy_from_slice(&data[src_offset..src_offset + unpadded_bytes_per_row as usize]);
        }
        drop(data);
        readback_buffer.unmap();
        Ok(rgba)
    }
}

fn create_output_texture(device: &Device, width: u32, height: u32) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("cell render output texture"),
        size: Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_storage_buffer_or_dummy<T: Pod>(device: &Device, label: &str, data: &[T]) -> Buffer {
    if data.is_empty() {
        return device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytes_of(&0u32),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
    }
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}
