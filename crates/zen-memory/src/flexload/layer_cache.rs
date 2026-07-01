use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Least Recently Used: evicts the layer that was used longest ago. (Poor for looping)
    Lru,
    /// Most Recently Used: evicts the layer that was just used. (Good for looping)
    Mru,
    /// Static Pinning: Keeps layers 0..K permanently pinned, streams the rest. (Best for FlexLoad)
    StaticPinning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchPolicy {
    Sequential,
    Windowed,
}

pub struct LayerCache<H, D> {
    pub vram_cache: HashMap<usize, D>,
    pub ram_cache: HashMap<usize, H>,
    pub vram_usage: u64,
    pub ram_usage: u64,
    vram_access_order: VecDeque<usize>,
    ram_access_order: VecDeque<usize>,
}

impl<H, D> LayerCache<H, D> {
    pub fn new() -> Self {
        Self {
            vram_cache: HashMap::new(),
            ram_cache: HashMap::new(),
            vram_usage: 0,
            ram_usage: 0,
            vram_access_order: VecDeque::new(),
            ram_access_order: VecDeque::new(),
        }
    }

    pub fn get_vram(&self, layer_id: usize) -> Option<&D> {
        self.vram_cache.get(&layer_id)
    }

    pub fn get_ram(&self, layer_id: usize) -> Option<&H> {
        self.ram_cache.get(&layer_id)
    }

    pub fn mark_vram_accessed(&mut self, layer_id: usize) {
        self.vram_access_order.retain(|&x| x != layer_id);
        self.vram_access_order.push_back(layer_id);
    }

    pub fn mark_ram_accessed(&mut self, layer_id: usize) {
        self.ram_access_order.retain(|&x| x != layer_id);
        self.ram_access_order.push_back(layer_id);
    }

    pub fn insert_vram(&mut self, layer_id: usize, layer: D, size_bytes: u64) {
        if self.vram_cache.insert(layer_id, layer).is_none() {
            self.vram_usage += size_bytes;
        }
        self.mark_vram_accessed(layer_id);
    }

    pub fn insert_ram(&mut self, layer_id: usize, layer: H, size_bytes: u64) {
        if self.ram_cache.insert(layer_id, layer).is_none() {
            self.ram_usage += size_bytes;
        }
        self.mark_ram_accessed(layer_id);
    }

    pub fn remove_vram(&mut self, layer_id: usize, size_bytes: u64) -> Option<D> {
        if let Some(layer) = self.vram_cache.remove(&layer_id) {
            self.vram_usage = self.vram_usage.saturating_sub(size_bytes);
            self.vram_access_order.retain(|&x| x != layer_id);
            Some(layer)
        } else {
            None
        }
    }

    pub fn remove_ram(&mut self, layer_id: usize, size_bytes: u64) -> Option<H> {
        if let Some(layer) = self.ram_cache.remove(&layer_id) {
            self.ram_usage = self.ram_usage.saturating_sub(size_bytes);
            self.ram_access_order.retain(|&x| x != layer_id);
            Some(layer)
        } else {
            None
        }
    }

    pub fn evict_vram_victim(&mut self, policy: CachePolicy, _current_layer_id: usize, plan_order: &[usize]) -> Option<usize> {
        match policy {
            CachePolicy::StaticPinning => {
                // Static Pinning: We want to keep the "earliest" layers in the plan pinned.
                // Evict the layer in cache that appears LAST in the plan order.
                self.vram_cache.keys()
                    .copied()
                    .max_by_key(|&id| plan_order.iter().position(|&x| x == id).unwrap_or(usize::MAX))
            }
            CachePolicy::Mru => self.vram_access_order.back().copied(),
            CachePolicy::Lru => self.vram_access_order.front().copied(),
        }
    }

    pub fn evict_ram_victim(&mut self, policy: CachePolicy, _current_layer_id: usize, plan_order: &[usize]) -> Option<usize> {
        match policy {
            CachePolicy::StaticPinning => {
                self.ram_cache.keys()
                    .copied()
                    .max_by_key(|&id| plan_order.iter().position(|&x| x == id).unwrap_or(usize::MAX))
            }
            CachePolicy::Mru => self.ram_access_order.back().copied(),
            CachePolicy::Lru => self.ram_access_order.front().copied(),
        }
    }
}
