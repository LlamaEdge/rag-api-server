pub(crate) mod ggml;

use crate::error;
use hyper::{Body, Request, Response};

pub(crate) async fn handle_llama_request(
    req: Request<Body>,
    chunk_capacity: usize,
) -> Response<Body> {
    match req.uri().path() {
        "/v1/chat/completions" => ggml::rag_query_handler(req).await,
        "/v1/models" => ggml::models_handler().await,
        "/v1/embeddings" => ggml::embeddings_handler(req).await,
        "/v1/files" => ggml::files_handler(req).await,
        "/v1/chunks" => ggml::chunks_handler(req).await,
        "/v1/retrieve" => ggml::retrieve_handler(req).await,
        "/v1/create/rag" => ggml::doc_to_embeddings_handler(req, chunk_capacity).await,
        "/v1/info" => ggml::server_info().await,
        _ => error::invalid_endpoint(req.uri().path()),
    }
}
