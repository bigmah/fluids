//! A compute device, shared with the renderer when there is one.
//!
//! The GPU solver runs on whatever device it is handed. In the app that is
//! Bevy's own, so buffers the solver fills can later be bound straight into
//! materials; tests open a headless one instead.

use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    Buffer, BufferBindingType, BufferSize, ComputePipeline, MapMode, PipelineLayoutDescriptor,
    PollType, RawComputePipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StorageTextureAccess, TextureFormat, TextureViewDimension,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};

/// What a compute pass binds at one slot.
#[derive(Clone, Copy, Debug)]
pub enum Binding {
    /// A uniform of this many bytes, optionally addressed by dynamic offset.
    Uniform { size: u64, dynamic: bool },
    /// A read-write storage buffer.
    Storage,
    /// A write-only 3D storage texture.
    VolumeOut(TextureFormat),
}

/// One compute entry point, with a layout naming exactly the bindings it uses.
pub struct Kernel {
    pub pipeline: ComputePipeline,
    pub layout: BindGroupLayout,
}

#[derive(Clone)]
pub struct Gpu {
    pub device: RenderDevice,
    pub queue: RenderQueue,
}

impl Gpu {
    pub fn new(device: RenderDevice, queue: RenderQueue) -> Self {
        Self { device, queue }
    }

    /// A device with no window, for tests and benchmarks. `None` on a machine
    /// without a usable adapter, so GPU tests can skip instead of failing.
    #[cfg(test)]
    pub fn headless() -> Option<Self> {
        Self::headless_with(wgpu::Features::empty())
    }

    /// [`Self::headless`], with optional features such as timestamp queries.
    #[cfg(test)]
    pub fn headless_with(features: wgpu::Features) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .ok()?;
        // Bevy asks for the adapter's full limits on native, so ask for the same.
        let (device, queue) =
            bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("fluids headless"),
                required_features: features & adapter.features(),
                required_limits: adapter.limits(),
                ..Default::default()
            }))
            .ok()?;
        Some(Self {
            device: RenderDevice::from(device),
            queue: RenderQueue(std::sync::Arc::new(
                bevy::render::renderer::WgpuWrapper::new(queue),
            )),
        })
    }

    pub fn module(&self, label: &str, source: &'static str) -> ShaderModule {
        self.device
            .create_and_validate_shader_module(ShaderModuleDescriptor {
                label: Some(label),
                source: ShaderSource::Wgsl(source.into()),
            })
    }

    pub fn kernel(
        &self,
        module: &ShaderModule,
        entry: &str,
        bindings: &[(u32, Binding)],
    ) -> Kernel {
        let entries: Vec<_> = bindings
            .iter()
            .map(|&(binding, kind)| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: match kind {
                    Binding::Uniform { size, dynamic } => BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: dynamic,
                        min_binding_size: BufferSize::new(size),
                    },
                    Binding::Storage => BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Binding::VolumeOut(format) => BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format,
                        view_dimension: TextureViewDimension::D3,
                    },
                },
                count: None,
            })
            .collect();
        let layout = self.device.create_bind_group_layout(entry, &entries);
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        Kernel { pipeline, layout }
    }

    pub fn bind_group(&self, kernel: &Kernel, entries: &[(u32, BindingResource<'_>)]) -> BindGroup {
        let entries: Vec<_> = entries
            .iter()
            .map(|(binding, resource)| BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect();
        self.device
            .create_bind_group(None, &kernel.layout, &entries)
    }

    /// Blocks until everything submitted so far has run.
    pub fn wait(&self) {
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("GPU device lost");
    }

    /// Maps a `MAP_READ` buffer, waits for it, and hands its bytes to `read`.
    pub fn read<T>(&self, buffer: &Buffer, read: impl FnOnce(&[u8]) -> T) -> T {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait();
        receiver
            .recv()
            .expect("the map callback never ran")
            .expect("failed to map a readback buffer");
        let out = read(&slice.get_mapped_range());
        buffer.unmap();
        out
    }
}

/// Reinterprets mapped bytes as plain data, copying only if the mapping is not
/// aligned for `T`.
pub fn cast<T: bytemuck::Pod>(bytes: &[u8]) -> std::borrow::Cow<'_, [T]> {
    match bytemuck::try_cast_slice(bytes) {
        Ok(slice) => std::borrow::Cow::Borrowed(slice),
        Err(_) => std::borrow::Cow::Owned(bytemuck::pod_collect_to_vec(bytes)),
    }
}
