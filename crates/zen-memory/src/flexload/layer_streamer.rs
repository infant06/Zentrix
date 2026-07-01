use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::thread;
use std::sync::mpsc::{self, Sender, Receiver};

use super::error::FlexLoadError;
use super::device_stage::{StageDevice, LayerState};
use super::memory_budget::{FlexLoadConfig, FlexLoadPlan};
use super::layer_cache::LayerCache;

pub trait LayerStore: Send + Sync {
    type LayerWeights: Send + Clone + 'static;

    fn layer_ids(&self) -> Vec<usize>;
    fn load_layer(&self, layer_id: usize) -> Result<Self::LayerWeights, FlexLoadError>;
    fn layer_size_bytes(&self, layer_id: usize) -> Result<u64, FlexLoadError>;
}

pub trait DeviceStager: Send + Sync {
    type HostLayer: Send + Clone + 'static;
    type DeviceLayer: Send + 'static;

    fn move_to_device(
        &self,
        layer: Self::HostLayer,
        device: StageDevice,
    ) -> Result<Self::DeviceLayer, FlexLoadError>;

    fn evict(&self, layer_id: usize) -> Result<(), FlexLoadError>;
}

struct LayerTask<L> {
    layer_id: usize,
    layer: L,
    size_bytes: u64,
}

pub struct LayerStreamer<S: LayerStore, D: DeviceStager<HostLayer = S::LayerWeights>> {
    #[allow(dead_code)]
    store: Arc<S>,
    device: Arc<D>,
    config: FlexLoadConfig,
    plan: FlexLoadPlan,

    // State tracking
    layer_states: HashMap<usize, LayerState>,
    cache: LayerCache<D::HostLayer, D::DeviceLayer>,
    
    // Async prefetch queue
    prefetch_queue: VecDeque<usize>,
    
    // Background thread channels
    request_tx: Sender<usize>,
    result_rx: Receiver<Result<LayerTask<S::LayerWeights>, FlexLoadError>>,
}

impl<S: LayerStore + 'static, D: DeviceStager<HostLayer = S::LayerWeights> + 'static> LayerStreamer<S, D> {
    pub fn new(store: Arc<S>, device: Arc<D>, config: FlexLoadConfig, plan: FlexLoadPlan) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<usize>();
        let (result_tx, result_rx) = mpsc::channel();

        // Start background prefetch thread
        let store_clone = store.clone();
        thread::spawn(move || {
            for layer_id in request_rx {
                let size_bytes = store_clone.layer_size_bytes(layer_id).unwrap_or(100_000_000);
                let load_result = store_clone.load_layer(layer_id);
                match load_result {
                    Ok(layer) => {
                        let _ = result_tx.send(Ok(LayerTask { layer_id, layer, size_bytes }));
                    }
                    Err(e) => {
                        let _ = result_tx.send(Err(e));
                    }
                }
            }
        });

        let mut streamer = Self {
            store,
            device,
            config,
            plan,
            layer_states: HashMap::new(),
            cache: LayerCache::new(),
            prefetch_queue: VecDeque::new(),
            request_tx,
            result_rx,
        };

        streamer.init_prefetch();
        streamer
    }

    fn init_prefetch(&mut self) {
        let schedule: Vec<usize> = self.plan.prefetch_schedule.iter().take(self.config.prefetch_ahead_count).copied().collect();
        for layer_id in schedule {
            self.queue_prefetch(layer_id);
        }
    }

    fn queue_prefetch(&mut self, layer_id: usize) {
        if self.layer_states.get(&layer_id) == Some(&LayerState::Queued) || self.layer_states.get(&layer_id) == Some(&LayerState::InRam) || self.layer_states.get(&layer_id) == Some(&LayerState::OnDevice) {
            return;
        }
        self.layer_states.insert(layer_id, LayerState::Queued);
        self.prefetch_queue.push_back(layer_id);
        let _ = self.request_tx.send(layer_id);
    }

    fn process_loaded_tasks(&mut self, current_layer_id: usize) -> Result<(), FlexLoadError> {
        while let Ok(result) = self.result_rx.try_recv() {
            match result {
                Ok(task) => {
                    self.layer_states.insert(task.layer_id, LayerState::InRam);
                    self.cache.insert_ram(task.layer_id, task.layer, task.size_bytes);
                    self.enforce_ram_budget(current_layer_id);
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    pub fn prepare_layer(&mut self, layer_id: usize) -> Result<(), FlexLoadError> {
        self.process_loaded_tasks(layer_id)?;

        if self.cache.get_vram(layer_id).is_some() {
            zen_telemetry::Telemetry::record_cache_hit(layer_id, "vram");
            self.cache.mark_vram_accessed(layer_id);
            self.layer_states.insert(layer_id, LayerState::OnDevice);
            return Ok(());
        }

        if self.cache.get_ram(layer_id).is_some() {
            zen_telemetry::Telemetry::record_cache_hit(layer_id, "ram");
        } else {
            zen_telemetry::Telemetry::record_cache_miss(layer_id);
            let state = self.layer_states.get(&layer_id);
            if state == Some(&LayerState::NotLoaded) || state == Some(&LayerState::Evicted) || state.is_none() {
                self.queue_prefetch(layer_id);
            }
            
            let start_fetch = std::time::Instant::now();
            loop {
                let result = self.result_rx.recv().map_err(|_| FlexLoadError::IoError("Channel closed".to_string()))?;
                match result {
                    Ok(task) => {
                        let tid = task.layer_id;
                        self.layer_states.insert(tid, LayerState::InRam);
                        self.cache.insert_ram(tid, task.layer, task.size_bytes);
                        self.enforce_ram_budget(layer_id);
                        if tid == layer_id {
                            zen_telemetry::Telemetry::record_layer_fetch_latency(layer_id, start_fetch.elapsed());
                            break;
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        let host_layer = self.cache.get_ram(layer_id).unwrap().clone();
        let size_bytes = self.store.layer_size_bytes(layer_id).unwrap_or(100_000_000);
        
        self.enforce_vram_budget_for(layer_id, size_bytes)?;
        
        self.layer_states.insert(layer_id, LayerState::MovingToDevice);
        let start_staging = std::time::Instant::now();
        let device_layer = self.device.move_to_device(host_layer, self.plan.target_device)?;
        zen_telemetry::Telemetry::record_layer_staging_latency(layer_id, start_staging.elapsed());
        
        self.cache.insert_vram(layer_id, device_layer, size_bytes);
        self.layer_states.insert(layer_id, LayerState::OnDevice);

        Ok(())
    }

    pub fn get_layer(&self, layer_id: usize) -> Option<&D::DeviceLayer> {
        self.cache.get_vram(layer_id)
    }

    pub fn release_layer(&mut self, layer_id: usize) -> Result<(), FlexLoadError> {
        let next_idx = self.plan.ordered_layer_ids.iter().position(|&id| id == layer_id).unwrap_or(0) + self.config.prefetch_ahead_count;
        if next_idx < self.plan.ordered_layer_ids.len() {
            let next_layer = self.plan.ordered_layer_ids[next_idx];
            self.queue_prefetch(next_layer);
        }
        Ok(())
    }

    fn enforce_vram_budget_for(&mut self, current_layer_id: usize, incoming_size: u64) -> Result<(), FlexLoadError> {
        while self.cache.vram_usage + incoming_size > self.config.max_vram_bytes {
            if let Some(victim) = self.cache.evict_vram_victim(self.config.eviction_policy, current_layer_id, &self.plan.ordered_layer_ids) {
                let size_bytes = self.store.layer_size_bytes(victim).unwrap_or(100_000_000);
                if self.cache.remove_vram(victim, size_bytes).is_some() {
                    let start_eviction = std::time::Instant::now();
                    self.device.evict(victim)?;
                    zen_telemetry::Telemetry::record_layer_eviction_latency(victim, start_eviction.elapsed());
                    self.layer_states.insert(victim, LayerState::InRam);
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    fn enforce_ram_budget(&mut self, current_layer_id: usize) {
        while self.cache.ram_usage > self.config.max_ram_bytes {
            if let Some(victim) = self.cache.evict_ram_victim(self.config.eviction_policy, current_layer_id, &self.plan.ordered_layer_ids) {
                let size_bytes = self.store.layer_size_bytes(victim).unwrap_or(100_000_000);
                if self.cache.remove_ram(victim, size_bytes).is_some() {
                    self.layer_states.insert(victim, LayerState::Evicted);
                }
            } else {
                break;
            }
        }
    }
}
