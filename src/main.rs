mod backend;
mod error;
mod utils;

use anyhow::Result;
use chat_prompts::{MergeRagContextPolicy, PromptTemplateType};
use clap::Parser;
use error::ServerError;
use hyper::{
    header,
    service::{make_service_fn, service_fn},
    Body, Request, Response, Server, StatusCode,
};
use llama_core::MetadataBuilder;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf};
use utils::{is_valid_url, log};

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

// global system prompt
pub(crate) static GLOBAL_RAG_PROMPT: OnceCell<String> = OnceCell::new();
// server info
pub(crate) static SERVER_INFO: OnceCell<ServerInfo> = OnceCell::new();

// default socket address
const DEFAULT_SOCKET_ADDRESS: &str = "0.0.0.0:8080";

#[derive(Clone, Debug)]
pub struct AppState {
    pub state_thing: String,
}

#[derive(Debug, Parser)]
#[command(name = "LlamaEdge-RAG API Server", version = env!("CARGO_PKG_VERSION"), author = env!("CARGO_PKG_AUTHORS"), about = "LlamaEdge-RAG API Server")]
struct Cli {
    /// Sets names for chat and embedding models. The names are separated by comma without space, for example, '--model-name Llama-2-7b,all-minilm'.
    #[arg(short, long, value_delimiter = ',', required = true)]
    model_name: Vec<String>,
    /// Model aliases for chat and embedding models
    #[arg(
        short = 'a',
        long,
        value_delimiter = ',',
        default_value = "default,embedding"
    )]
    model_alias: Vec<String>,
    /// Sets context sizes for chat and embedding models. The sizes are separated by comma without space, for example, '--ctx-size 4096,384'. The first value is for the chat model, and the second is for the embedding model.
    #[arg(
        short = 'c',
        long,
        value_delimiter = ',',
        default_value = "4096,384",
        value_parser = clap::value_parser!(u64)
    )]
    ctx_size: Vec<u64>,
    /// Prompt template.
    #[arg(short, long, value_parser = clap::value_parser!(PromptTemplateType), required = true)]
    prompt_template: PromptTemplateType,
    /// Halt generation at PROMPT, return control.
    #[arg(short, long)]
    reverse_prompt: Option<String>,
    /// Batch size for prompt processing
    #[arg(short, long, default_value = "512")]
    batch_size: u64,
    /// Custom rag prompt.
    #[arg(long)]
    rag_prompt: Option<String>,
    /// Strategy for merging RAG context into chat messages.
    #[arg(long = "rag-policy", default_value_t, value_enum)]
    policy: MergeRagContextPolicy,
    /// URL of Qdrant REST Service
    #[arg(long, default_value = "http://localhost:6333")]
    qdrant_url: String,
    /// Name of Qdrant collection
    #[arg(long, default_value = "default")]
    qdrant_collection_name: String,
    /// Max number of retrieved result (no less than 1)
    #[arg(long, default_value = "5", value_parser = clap::value_parser!(u64))]
    qdrant_limit: u64,
    /// Minimal score threshold for the search result
    #[arg(long, default_value = "0.4", value_parser = clap::value_parser!(f32))]
    qdrant_score_threshold: f32,
    /// Maximum number of tokens each chunk contains
    #[arg(long, default_value = "100", value_parser = clap::value_parser!(usize))]
    chunk_capacity: usize,
    /// Print prompt strings to stdout
    #[arg(long)]
    log_prompts: bool,
    /// Print statistics to stdout
    #[arg(long)]
    log_stat: bool,
    /// Print all log information to stdout
    #[arg(long)]
    log_all: bool,
    /// Socket address of LlamaEdge API Server instance
    #[arg(long, default_value = DEFAULT_SOCKET_ADDRESS)]
    socket_addr: String,
    /// Root path for the Web UI files
    #[arg(long, default_value = "chatbot-ui")]
    web_ui: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), ServerError> {
    let cli = Cli::parse();

    // log the version of the server
    let server_version = env!("CARGO_PKG_VERSION").to_string();
    log(format!(
        "\n[INFO] LlamaEdge-RAG version: {}",
        &server_version
    ));

    // log the cli options
    if cli.model_name.len() != 2 {
        return Err(ServerError::ArgumentError(
            "LlamaEdge RAG API server requires a chat model and an embedding model.".to_owned(),
        ));
    }
    log(format!(
        "[INFO] Model names: {names}",
        names = &cli.model_name.join(",")
    ));
    if cli.model_alias.len() != 2 {
        return Err(ServerError::ArgumentError(
            "LlamaEdge RAG API server requires two model aliases: one for chat model, one for embedding model.".to_owned(),
        ));
    }
    log(format!(
        "[INFO] Model aliases: {aliases}",
        aliases = &cli.model_alias.join(",")
    ));
    if cli.ctx_size.len() != 2 {
        return Err(ServerError::ArgumentError(
            "LlamaEdge RAG API server requires two context sizes: one for chat model, one for embedding model.".to_owned(),
        ));
    }
    let ctx_sizes_str: String = cli
        .ctx_size
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<String>>()
        .join(",");
    log(format!(
        "[INFO] Context sizes: {ctx_sizes}",
        ctx_sizes = ctx_sizes_str
    ));
    log(format!("[INFO] Prompt template: {}", &cli.prompt_template));
    if let Some(reverse_prompt) = &cli.reverse_prompt {
        log(format!("[INFO] reverse prompt: {}", reverse_prompt));
    }

    if let Some(rag_prompt) = &cli.rag_prompt {
        log(format!("[INFO] rag prompt: {}", rag_prompt));
        GLOBAL_RAG_PROMPT.set(rag_prompt.clone()).map_err(|_| {
            ServerError::Operation("Failed to set `GLOBAL_SYSTEM_PROMPT`.".to_string())
        })?;
    }

    if !is_valid_url(&cli.qdrant_url) {
        return Err(ServerError::ArgumentError(format!(
            "The URL of Qdrant REST API is invalid: {}.",
            &cli.qdrant_url
        )));
    }
    log(format!("[INFO] Qdrant server url: {}", &cli.qdrant_url));
    log(format!(
        "[INFO] Qdrant collection name: {}",
        &cli.qdrant_collection_name
    ));
    log(format!(
        "[INFO] Max number of retrieved result: {}",
        &cli.qdrant_limit
    ));
    log(format!(
        "[INFO] Qdrant score threshold: {}",
        &cli.qdrant_score_threshold
    ));
    let qdrant_config = QdrantConfig {
        url: cli.qdrant_url,
        collection_name: cli.qdrant_collection_name,
        limit: cli.qdrant_limit,
        score_threshold: cli.qdrant_score_threshold,
    };

    log(format!(
        "[INFO] Chunk capacity (in tokens): {}",
        &cli.chunk_capacity
    ));
    log(format!("[INFO] Enable prompt log: {}", &cli.log_prompts));
    log(format!("[INFO] Enable plugin log: {}", &cli.log_stat));
    log(format!("[INFO] Socket address: {}", &cli.socket_addr));

    // RAG policy
    let mut policy = cli.policy;
    log(format!("[INFO] RAG policy: {}", policy));
    if policy == MergeRagContextPolicy::SystemMessage && !cli.prompt_template.has_system_prompt() {
        println!("       * [WARINING] The chat model does not support system message, while the '--policy' option sets to \"{}\". Update the RAG policy to {}.", cli.policy, MergeRagContextPolicy::LastUserMessage);
        policy = MergeRagContextPolicy::LastUserMessage;
        log(format!("       * Updated RAG policy: {}", policy));
    }

    // create metadata for chat model
    let chat_metadata = MetadataBuilder::new(
        cli.model_name[0].clone(),
        cli.model_alias[0].clone(),
        cli.prompt_template,
    )
    .with_ctx_size(cli.ctx_size[0])
    .with_reverse_prompt(cli.reverse_prompt)
    .with_batch_size(cli.batch_size)
    .enable_prompts_log(cli.log_prompts || cli.log_all)
    .enable_plugin_log(cli.log_stat || cli.log_all)
    .build();

    let chat_model_info = ModelConfig {
        name: chat_metadata.model_name.clone(),
        ty: "chat".to_string(),
        prompt_template: chat_metadata.prompt_template,
        n_predict: chat_metadata.n_predict,
        reverse_prompt: chat_metadata.reverse_prompt.clone(),
        n_gpu_layers: chat_metadata.n_gpu_layers,
        ctx_size: chat_metadata.ctx_size,
        batch_size: chat_metadata.batch_size,
        temperature: chat_metadata.temperature,
        top_p: chat_metadata.top_p,
        repeat_penalty: chat_metadata.repeat_penalty,
        presence_penalty: chat_metadata.presence_penalty,
        frequency_penalty: chat_metadata.frequency_penalty,
    };

    // chat model
    let chat_models = [chat_metadata];

    // create metadata for embedding model
    let embedding_metadata = MetadataBuilder::new(
        cli.model_name[1].clone(),
        cli.model_alias[1].clone(),
        cli.prompt_template,
    )
    .with_ctx_size(cli.ctx_size[1])
    .with_batch_size(cli.batch_size)
    .enable_prompts_log(cli.log_prompts || cli.log_all)
    .enable_plugin_log(cli.log_stat || cli.log_all)
    .build();

    let embedding_model_info = ModelConfig {
        name: embedding_metadata.model_name.clone(),
        ty: "embedding".to_string(),
        prompt_template: embedding_metadata.prompt_template,
        n_predict: embedding_metadata.n_predict,
        reverse_prompt: embedding_metadata.reverse_prompt.clone(),
        n_gpu_layers: embedding_metadata.n_gpu_layers,
        ctx_size: embedding_metadata.ctx_size,
        batch_size: embedding_metadata.batch_size,
        temperature: embedding_metadata.temperature,
        top_p: embedding_metadata.top_p,
        repeat_penalty: embedding_metadata.repeat_penalty,
        presence_penalty: embedding_metadata.presence_penalty,
        frequency_penalty: embedding_metadata.frequency_penalty,
    };

    // embedding model
    let embedding_models = [embedding_metadata];

    // create rag config
    let rag_config = RagConfig {
        chat_model: chat_model_info,
        embedding_model: embedding_model_info,
        policy: cli.policy,
    };

    // initialize the core context
    llama_core::init_rag_core_context(&chat_models[..], &embedding_models[..]).map_err(|e| {
        ServerError::Operation(format!("Failed to initialize the core context. {}", e))
    })?;

    // get the plugin version info
    let plugin_info =
        llama_core::get_plugin_info().map_err(|e| ServerError::Operation(e.to_string()))?;
    let plugin_version = format!(
        "b{build_number} (commit {commit_id})",
        build_number = plugin_info.build_number,
        commit_id = plugin_info.commit_id,
    );
    log(format!("[INFO] Wasi-nn-ggml plugin: {}", &plugin_version));

    // socket address
    let addr = cli
        .socket_addr
        .parse::<SocketAddr>()
        .map_err(|e| ServerError::SocketAddr(e.to_string()))?;

    // set the server info
    let port = addr.port().to_string();

    // create server info
    let server_info = ServerInfo {
        version: server_version,
        plugin_version,
        port,
        rag_config,
        qdrant_config,
    };
    SERVER_INFO
        .set(server_info)
        .map_err(|_| ServerError::Operation("Failed to set `SERVER_INFO`.".to_string()))?;

    let new_service = make_service_fn(move |_| {
        let web_ui = cli.web_ui.to_string_lossy().to_string();
        let chunk_capacity = cli.chunk_capacity;

        async move {
            Ok::<_, Error>(service_fn(move |req| {
                handle_request(req, chunk_capacity, web_ui.clone())
            }))
        }
    });
    let server = Server::bind(&addr).serve(new_service);
    log(format!(
        "[INFO] LlamaEdge-RAG API server listening on http://{}:{}",
        addr.ip(),
        addr.port()
    ));

    match server.await {
        Ok(_) => Ok(()),
        Err(e) => Err(ServerError::Operation(e.to_string())),
    }
}

async fn handle_request(
    req: Request<Body>,
    chunk_capacity: usize,
    web_ui: String,
) -> Result<Response<Body>, hyper::Error> {
    let path_str = req.uri().path();
    let path_buf = PathBuf::from(path_str);
    let mut path_iter = path_buf.iter();
    path_iter.next(); // Must be Some(OsStr::new(&path::MAIN_SEPARATOR.to_string()))
    let root_path = path_iter.next().unwrap_or_default();
    let root_path = "/".to_owned() + root_path.to_str().unwrap_or_default();

    match root_path.as_str() {
        "/echo" => Ok(Response::new(Body::from("echo test"))),
        "/v1" => backend::handle_llama_request(req, chunk_capacity).await,
        _ => Ok(static_response(path_str, web_ui)),
    }
}

fn static_response(path_str: &str, root: String) -> Response<Body> {
    let path = match path_str {
        "/" => "/index.html",
        _ => path_str,
    };

    let mime = mime_guess::from_path(path);

    match std::fs::read(format!("{root}/{path}")) {
        Ok(content) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.first_or_text_plain().to_string())
            .body(Body::from(content))
            .unwrap(),
        Err(_) => {
            let body = Body::from(std::fs::read(format!("{root}/404.html")).unwrap_or_default());
            Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header(header::CONTENT_TYPE, "text/html")
                .body(body)
                .unwrap()
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct QdrantConfig {
    pub(crate) url: String,
    pub(crate) collection_name: String,
    pub(crate) limit: u64,
    pub(crate) score_threshold: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ModelConfig {
    // model name
    name: String,
    // type: chat or embedding
    #[serde(rename = "type")]
    ty: String,
    pub prompt_template: PromptTemplateType,
    pub n_predict: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reverse_prompt: Option<String>,
    pub n_gpu_layers: u64,
    pub ctx_size: u64,
    pub batch_size: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub repeat_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ServerInfo {
    version: String,
    plugin_version: String,
    port: String,
    // models: Vec<ModelConfig>,
    rag_config: RagConfig,
    qdrant_config: QdrantConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RagConfig {
    pub chat_model: ModelConfig,
    pub embedding_model: ModelConfig,
    pub policy: MergeRagContextPolicy,
}
