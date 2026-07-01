use anyhow::anyhow;
use axum::{
    extract::{Json, State},
    http,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};

use utoipa::ToSchema;

use crate::{
    handler_core::{ErrorToResponse, JsonError},
    types::ExtractedZentrixState,
    util::sanitize_error_message,
};

#[derive(Debug, Deserialize, ToSchema)]
pub struct VectorSearchRequest {
    pub model: Option<String>,
    #[schema(value_type = Vec<f32>)]
    pub query: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ApiVectorSearchHit {
    pub id: String,
    pub score: f32,
    #[schema(value_type = Object)]
    pub metadata: serde_json::Value,
    #[schema(value_type = Option<Vec<f32>>)]
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct VectorSearchResponse {
    pub hits: Vec<ApiVectorSearchHit>,
}

pub enum VectorSearchResponder {
    Json(VectorSearchResponse),
    InternalError(anyhow::Error),
}

impl IntoResponse for VectorSearchResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            VectorSearchResponder::Json(s) => Json(s).into_response(),
            VectorSearchResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.root_cause()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

#[utoipa::path(
    post,
    tag = "Zentrix",
    path = "/v1/zentrix/vector/search",
    request_body = VectorSearchRequest,
    responses(
        (status = 200, description = "Vector search hits", body = VectorSearchResponse),
    )
)]
pub async fn vector_search(
    State(state): ExtractedZentrixState,
    Json(req): Json<VectorSearchRequest>,
) -> impl IntoResponse {
    let hits = match state.vector_search(req.model.as_deref(), &req.query, req.top_k).await {
        Ok(h) => h.into_iter().map(|h| ApiVectorSearchHit {
            id: h.id,
            score: h.score,
            metadata: h.metadata,
            vector: h.vector,
        }).collect(),
        Err(e) => return VectorSearchResponder::InternalError(anyhow!(e)).into_response(),
    };

    VectorSearchResponder::Json(VectorSearchResponse { hits }).into_response()
}
