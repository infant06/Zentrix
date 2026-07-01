use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use zen_core::request::Request;
use crate::queue::RequestQueue;

#[derive(Clone)]
pub struct GlobalScheduler {
    queue: Arc<RequestQueue>,
}

impl GlobalScheduler {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(RequestQueue::new()),
        }
    }

    pub fn queue(&self) -> Arc<RequestQueue> {
        self.queue.clone()
    }

    /// Background task that continuously polls the queue and routes to available engines
    pub async fn run(&self, engines: Vec<Sender<Request>>) {
        loop {
            if let Some(req) = self.queue.dequeue().await {
                // Simplest round-robin or load-balancing for now
                if let Some(engine) = engines.first() {
                    let _ = engine.send(req).await;
                }
            } else {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        }
    }
}
