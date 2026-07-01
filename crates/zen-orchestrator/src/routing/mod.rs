use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDefinition {
    pub name: String,
    pub description: String,
    pub target_model_id: String,
    // Vector of text examples that represent this route
    pub examples: Vec<String>,
}

#[derive(Default, Clone)]
pub struct SemanticRouter {
    pub routes: Vec<RouteDefinition>,
}

impl SemanticRouter {
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    pub fn add_route(&mut self, route: RouteDefinition) {
        self.routes.push(route);
    }

    /// Lightweight routing based on keywords and exact string matches.
    /// In advanced implementations, this could use `EmbeddingIndexProvider` to embed the `query`
    /// and find the closest matching example vector.
    pub fn route_query(&self, query: &str) -> Option<String> {
        // Keyword matching: find a route whose description or examples contain query keywords
        let query_lower = query.to_lowercase();
        for route in &self.routes {
            if route.name.to_lowercase() == query_lower {
                return Some(route.target_model_id.clone());
            }
            for example in &route.examples {
                if query_lower.contains(&example.to_lowercase()) {
                    return Some(route.target_model_id.clone());
                }
            }
        }
        
        // Default to the first route if any
        self.routes.first().map(|r| r.target_model_id.clone())
    }
}
