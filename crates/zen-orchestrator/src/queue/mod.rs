use std::collections::VecDeque;
use tokio::sync::Mutex;
use zen_core::request::Request;

#[derive(Debug, Default)]
pub struct RequestQueue {
    queue: Mutex<VecDeque<Request>>,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    pub async fn enqueue(&self, req: Request) {
        let mut q = self.queue.lock().await;
        q.push_back(req);
    }

    pub async fn dequeue(&self) -> Option<Request> {
        let mut q = self.queue.lock().await;
        q.pop_front()
    }

    pub async fn len(&self) -> usize {
        let q = self.queue.lock().await;
        q.len()
    }
}
