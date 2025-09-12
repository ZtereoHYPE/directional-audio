use std::ops::AddAssign;
use crate::audio_engine::gpu_constants::{MAX_INSTANCES, SPHERE_POINTS};
use crate::audio_engine::rays::RtOutputBufferData;
use crate::audio_engine::{AudioInstance, InstanceBufferData, RtSyncStage};
use crate::vulkan::buffer::{InlineBufferData, LocalVulkanBuffer, VulkanBuffer};
use approx_dbscan::approximate_dbscan;
use approx_dbscan::clusterable::{Clusterable, Point};
use ash::vk::{AccessFlags, Buffer, BufferUsageFlags, CommandBuffer, DependencyFlags, MemoryBarrier, PipelineStageFlags, Semaphore, SemaphoreSignalInfo, SemaphoreWaitInfo};
use ash::Device;
use glam::Vec3;
use std::rc::Rc;
use std::slice::from_ref;
use std::sync::{mpsc, Arc, Mutex};
use std::sync::mpsc::Sender;
use std::thread;
use std::thread::{spawn, JoinHandle, Thread};
use std::time::{Duration, Instant};
use bytemuck::Zeroable;
use vk_mem::Allocator;
use crate::vulkan::in_flight::{InFlightCounter};
use crate::vulkan::timeline::TimelineTracker;

// todo: move the task from a lambda to a function pointer of strict type to avoid dynamic dispatching!
type Task = Box<dyn Fn(&mut LocalVulkanBuffer<RtOutputBufferData>, &mut LocalVulkanBuffer<InstanceBufferData>) + Send>;

pub(super) struct ClusterModule {
    device: Device,
    rt_output_handle: Buffer,

    local_rt_output_buffer_handle: Buffer,
    local_instance_buffer_handle: Buffer,

    pub(super) instance_buffer: VulkanBuffer<InstanceBufferData>,

    work_thread: JoinHandle<()>,
    work_queue: Sender<Task>,

    pub(super) last_clusters: Vec<AudioInstance>,
    in_progress_clusters: Arc<Mutex<Vec<AudioInstance>>>,
}

impl ClusterModule {
    pub(super) fn new(
        allocator: Arc<Allocator>,
        device: Device,
        rt_output_buffer: &VulkanBuffer<RtOutputBufferData>,
    ) -> Self {
        let instance_buffer: VulkanBuffer<InstanceBufferData> = VulkanBuffer::new_inline(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST,
            allocator.clone()
        );
        
        let (work_queue, work_receiver) = mpsc::channel();
        let (handle_sender, handle_receiver) = mpsc::channel();

        let work_thread = thread::spawn(move || {
            let mut local_rt_output_buffer: LocalVulkanBuffer<RtOutputBufferData> = LocalVulkanBuffer::new_inline(
                BufferUsageFlags::TRANSFER_DST,
                allocator.clone()
            );

            let mut local_instance_buffer: LocalVulkanBuffer<InstanceBufferData> = LocalVulkanBuffer::new_inline(
                BufferUsageFlags::TRANSFER_SRC,
                allocator.clone()
            );


            handle_sender.send(local_rt_output_buffer.handle());
            handle_sender.send(local_instance_buffer.handle());

            loop {
                let mut task: Task = work_receiver.recv().unwrap();
                task(&mut local_rt_output_buffer, &mut local_instance_buffer);
            }
        });

        let local_rt_output_buffer_handle = handle_receiver.recv().unwrap();
        let local_instance_buffer_handle = handle_receiver.recv().unwrap();

        Self {
            device,
            local_rt_output_buffer_handle,
            local_instance_buffer_handle,
            rt_output_handle: rt_output_buffer.handle(),
            last_clusters: vec![],
            in_progress_clusters: Arc::new(Mutex::new(vec![])),
            work_thread,
            work_queue,
            instance_buffer
        }
    }

    pub(super) fn copy_rt_output(&mut self, command_buffer: &mut CommandBuffer) {
        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::TRANSFER_READ); // invalidate any shader read caches

        unsafe {
            self.device.cmd_pipeline_barrier(
                *command_buffer,
                PipelineStageFlags::COMPUTE_SHADER, // wait for all compute dispatches so far...
                PipelineStageFlags::TRANSFER, // ...before executing any transfer from now on
                DependencyFlags::empty(),
                from_ref(&memory_barrier),
                &[],
                &[]
            );

            self.device.cmd_copy_buffer(
                *command_buffer,
                self.rt_output_handle,
                self.local_rt_output_buffer_handle,
                RtOutputBufferData::region()
            )
        }
    }

    // todo: a lot of the work in the lambda can be moved to the cluster function, and the thread can
    //  simply execute a function pointer!
    pub(super) fn cluster_hits_async(&mut self, timeline: TimelineTracker<RtSyncStage>, counter: InFlightCounter) {
        let device = self.device.clone();
        let new_clusters = self.in_progress_clusters.clone();

        self.work_queue.send(Box::new(move |
            rt_output: &mut LocalVulkanBuffer<RtOutputBufferData>,
            instance_buf: &mut LocalVulkanBuffer<InstanceBufferData>
        | {
            let (wait_semaphore, wait_value) = timeline.get_wait_info(counter, RtSyncStage::Cluster);
            let wait_info = SemaphoreWaitInfo::default()
                .semaphores(&wait_semaphore)
                .values(&wait_value);

            let (finish_semaphore, finish_value) = timeline.get_signal_info(counter, RtSyncStage::Cluster);
            let finished_info = SemaphoreSignalInfo::default()
                .semaphore(finish_semaphore[0])
                .value(finish_value[0]);

            unsafe {
                device.wait_semaphores(&wait_info, u64::MAX);
                let initial_time = Instant::now();

                // todo: look for ways to speed this up, perhaps changing the allocation flags?
                let buffer_data = rt_output.buffer_data().to_local_copy();

                let copy_time = Instant::now();

                let instances = ClusterModule::cluster(buffer_data);

                let cluster_time = Instant::now();

                instance_buf.buffer_data().copy_instances(&instances);
                instance_buf.flush();
                *new_clusters.lock().unwrap() = instances;

                device.signal_semaphore(&finished_info);
                println!("Copy time: {:?}, Clustering time: {:?}, copy time 2: {:?}", copy_time - initial_time, cluster_time - copy_time, cluster_time.elapsed());
            }
        }));
    }

    unsafe fn cluster(rt_output: Box<RtOutputBufferData>) -> Vec<AudioInstance> {
        // Split the different sources' hits, filtering to only the ones that reached the source
        let source_clusters = rt_output.outputs
            .as_chunks_unchecked::<SPHERE_POINTS>() // split the various sources
            .iter()
            .map(|chunk|
                chunk.iter()
                    .filter(|ray| ray.found_source) // filter each chunk to only valid outputs
                    .map(|o| AudioClusterPoint {
                        direction: o.direction,
                        distance: o.additional_distance,
                        attenuation: o.attenuation,
                    })
                    .collect::<Vec<_>>() // collect each chunk into a vector of audio instances
            )
            .enumerate()
            .filter(|(_, src)| !src.is_empty()) // remove empty sources
            .map(|(idx, src)| (idx, src.len(), approximate_dbscan(src, 2.5, 0.2, 1))) // cluster them
            // .map(|(idx, src)| (idx, src.len(), src.into_iter().map(|i| vec![i]).collect::<Vec<_>>())) // cluster them
            .collect::<Vec<_>>();

        // todo: this is very easily parallelizable or SIMD-able
        let mut total_instances = vec![];
        for (source_idx, source_len, source) in source_clusters {
            for cluster in source.iter().skip(1) {
                let mut avg = AudioClusterPoint::zeroed();

                for point in cluster {
                    avg += point.clone();
                }

                let cluster_size = cluster.len() as u32;

                let cluster_size_attenuation = cluster_size as f32 / source_len as f32;
                total_instances.push(AudioInstance {
                    direction: avg.direction / cluster_size as f32,
                    distance: avg.distance / cluster_size as f32,
                    attenuation: (avg.attenuation / cluster_size as f32) * cluster_size_attenuation,
                    cluster_size,
                    index: source_idx as u32,
                });
            }
        }

        // If we have too many instances, only take the most important ones
        if total_instances.len() > MAX_INSTANCES {
            total_instances.sort_by(|l, r| l.cluster_size.cmp(&r.cluster_size));
            total_instances.truncate(MAX_INSTANCES);
        }

        total_instances
    }

    pub(super) fn instance_amt(&self) -> usize {
        self.last_clusters.len()
    }

    pub(crate) unsafe fn upload_to_buffer(&mut self, command_buffer: &mut CommandBuffer) {
        self.device.cmd_copy_buffer(
            *command_buffer,
            self.local_instance_buffer_handle,
            self.instance_buffer.handle(),
            InstanceBufferData::region()
        );

        let memory_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::TRANSFER_WRITE) // flush any transfer write caches
            .dst_access_mask(AccessFlags::SHADER_READ); // invalidate any shader read caches

        self.device.cmd_pipeline_barrier(
            *command_buffer,
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            from_ref(&memory_barrier),
            &[],
            &[]
        )
    }
    
    /// Sets the last clusters to the in-progress (and now completed) ones
    pub(super) fn update_instances(&mut self) {
        self.last_clusters = self.in_progress_clusters.lock().unwrap().clone()
    }
}

impl Drop for ClusterModule {
    fn drop(&mut self) {
        // todo: terminate thread on drop
    }
}

#[derive(Clone, Zeroable)]
struct AudioClusterPoint {
    direction: Vec3,
    distance: f32,
    attenuation: f32,
}

impl Clusterable<3> for AudioClusterPoint {
    fn distance(&self, other: &Self) -> f64 {
        self.direction.distance(other.direction) as f64
    }

    fn point(&self) -> Point<3> {
        [
            self.direction.x as f64,
            self.direction.y as f64,
            self.direction.z as f64,
        ]
    }

    fn nth(&self, idx: usize) -> f64 {
        match idx {
            0 => self.direction.x as f64,
            1 => self.direction.y as f64,
            2 => self.direction.z as f64,
            _ => unreachable!()
        }
    }
}

impl AddAssign for AudioClusterPoint {
    fn add_assign(&mut self, rhs: Self) {
        self.attenuation += rhs.attenuation;
        self.distance += rhs.distance;
        self.direction += rhs.direction;
    }
}
