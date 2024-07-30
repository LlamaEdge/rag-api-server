use crate::{error, utils::gen_chat_id, GLOBAL_RAG_PROMPT, SERVER_INFO};
use chat_prompts::{error as ChatPromptsError, MergeRagContext, MergeRagContextPolicy};
use endpoints::{
    chat::{ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionUserMessageContent},
    embeddings::EmbeddingRequest,
    files::{DeleteFileStatus, FileObject, ListFilesResponse},
    rag::{ChunksRequest, ChunksResponse, RagEmbeddingRequest, RetrieveObject},
};
use futures_util::TryStreamExt;
use hyper::{body::to_bytes, Body, Method, Request, Response};
use multipart::server::{Multipart, ReadEntry, ReadEntryResult};
use multipart_2021 as multipart;
use std::{
    fs::{self, File},
    io::{Cursor, Read, Write},
    path::Path,
    time::SystemTime,
};
use walkdir::{DirEntry, WalkDir};

/// List all models available.
pub(crate) async fn models_handler() -> Response<Body> {
    // log
    info!(target: "models_handler", "Handling the coming model list request.");

    let list_models_response = match llama_core::models::models().await {
        Ok(list_models_response) => list_models_response,
        Err(e) => {
            let err_msg = format!("Failed to get model list. Reason: {}", e);

            // log
            error!(target: "models_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // serialize response
    let s = match serde_json::to_string(&list_models_response) {
        Ok(s) => s,
        Err(e) => {
            let err_msg = format!("Failed to serialize the model list result. Reason: {}", e);

            // log
            error!(target: "models_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .header("Content-Type", "application/json")
        .body(Body::from(s));
    let res = match result {
        Ok(response) => response,
        Err(e) => {
            let err_msg = format!("Failed to get model list. Reason: {}", e);

            // log
            error!(target: "models_handler", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    // log
    info!(target: "models_handler", "Send the model list response.");

    res
}

/// Compute embeddings for the input text and return the embeddings object.
pub(crate) async fn embeddings_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "embeddings_handler", "Handling the coming embeddings request");

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "embeddings_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut embedding_request: EmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(embedding_request) => embedding_request,
        Err(e) => {
            let err_msg = format!("Fail to deserialize embedding request: {msg}", msg = e);

            // log
            error!(target: "embeddings_handler", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    if embedding_request.user.is_none() {
        embedding_request.user = Some(gen_chat_id())
    };
    let id = embedding_request.user.clone().unwrap();

    // log user id
    info!(target: "embedding_request", "user: {}", &id);

    let res = match llama_core::embeddings::embeddings(&embedding_request).await {
        Ok(embedding_response) => {
            // serialize embedding object
            match serde_json::to_string(&embedding_response) {
                Ok(s) => {
                    // return response
                    let result = Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .header("Content-Type", "application/json")
                        .header("user", id)
                        .body(Body::from(s));
                    match result {
                        Ok(response) => response,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "embeddings_handler", "{}", &err_msg);

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize embedding object. {}", e);

                    // log
                    error!(target: "embeddings_handler", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "embeddings_handler", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "embeddings_handler", "Send the embeddings response");

    res
}

/// Query a user input and return a chat-completion response with the answer from the model.
///
/// Note that the body of the request is deserialized to a `ChatCompletionRequest` instance.
pub(crate) async fn rag_query_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "rag_query_handler", "Handling the coming rag query request");

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "rag_query_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    info!(target: "rag_query_handler", "Prepare the chat completion request.");

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "rag_query_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            let err_msg = format!("Fail to deserialize chat completion request: {}", e);

            // log
            error!(target: "rag_query_handler", "{}", &err_msg);

            // log body_bytes
            error!(target: "rag_query_handler", "raw data:\n{:?}", &body_bytes.to_ascii_lowercase());

            return error::bad_request(err_msg);
        }
    };

    // check if the user id is provided
    if chat_request.user.is_none() {
        chat_request.user = Some(gen_chat_id())
    };
    let id = chat_request.user.clone().unwrap();

    // log user id
    info!(target: "rag_query_handler", "user: {}", &id);

    let server_info = match SERVER_INFO.get() {
        Some(server_info) => server_info,
        None => {
            let err_msg = "The server info is not set.";

            // log
            error!(target: "rag_query_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    info!(target: "rag_query_handler", "Compute embeddings for user query.");

    // * compute embeddings for user query
    let embedding_response = match chat_request.messages.is_empty() {
        true => {
            let err_msg = "Messages should not be empty.";

            // log
            error!(target: "rag_query_handler", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
        false => {
            let last_message = chat_request.messages.last().unwrap();
            match last_message {
                ChatCompletionRequestMessage::User(user_message) => {
                    let query_text = match user_message.content() {
                        ChatCompletionUserMessageContent::Text(text) => text,
                        _ => {
                            let err_msg = "The last message must be a text content user message";

                            // log
                            error!(target: "rag_query_handler", "{}", &err_msg);

                            return error::bad_request(err_msg);
                        }
                    };

                    // log
                    info!(target: "rag_query_handler", "query text: {}", query_text);

                    // get the available embedding models
                    let embedding_model_names = match llama_core::utils::embedding_model_names() {
                        Ok(model_names) => model_names,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "rag_query_handler", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    };

                    // create a embedding request
                    let embedding_request = EmbeddingRequest {
                        model: embedding_model_names[0].clone(),
                        input: query_text.into(),
                        encoding_format: None,
                        user: chat_request.user.clone(),
                    };

                    let rag_embedding_request = RagEmbeddingRequest {
                        embedding_request,
                        qdrant_url: server_info.qdrant_config.url.clone(),
                        qdrant_collection_name: server_info.qdrant_config.collection_name.clone(),
                    };

                    // compute embeddings for query
                    match llama_core::rag::rag_query_to_embeddings(&rag_embedding_request).await {
                        Ok(embedding_response) => embedding_response,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "rag_query_handler", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    }
                }
                _ => {
                    let err_msg = "The last message must be a user message";

                    // log
                    error!(target: "rag_query_handler", "{}", &err_msg);

                    return error::bad_request(err_msg);
                }
            }
        }
    };
    let query_embedding: Vec<f32> = match embedding_response.data.first() {
        Some(embedding) => embedding.embedding.iter().map(|x| *x as f32).collect(),
        None => {
            let err_msg = "No embeddings returned";

            // log
            error!(target: "rag_query_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // * retrieve context
    let res = match llama_core::rag::rag_retrieve_context(
        query_embedding.as_slice(),
        server_info.qdrant_config.url.to_string().as_str(),
        server_info.qdrant_config.collection_name.as_str(),
        server_info.qdrant_config.limit as usize,
        Some(server_info.qdrant_config.score_threshold),
    )
    .await
    {
        Ok(search_result) => Some(search_result),
        Err(e) => {
            // log
            error!(target: "rag_query_handler", "No point retrieved. {}", e);

            None
        }
    };

    if let Some(ro) = res {
        match ro.points {
            Some(scored_points) => {
                match scored_points.is_empty() {
                    true => {
                        // log
                        warn!(target: "rag_query_handler", "{}", format!("No point retrieved (score < threshold {})", server_info.qdrant_config.score_threshold));
                    }
                    false => {
                        // update messages with retrieved context
                        let mut context = String::new();
                        for (idx, point) in scored_points.iter().enumerate() {
                            // log
                            info!(target: "rag_query_handler", "point: {}, score: {}, source: {}", idx, point.score, &point.source);

                            context.push_str(&point.source);
                            context.push_str("\n\n");
                        }

                        if chat_request.messages.is_empty() {
                            let err_msg = "No message in the chat request.";

                            // log
                            error!(target: "rag_query_handler", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }

                        let prompt_template = match llama_core::utils::chat_prompt_template(
                            chat_request.model.as_deref(),
                        ) {
                            Ok(prompt_template) => prompt_template,
                            Err(e) => {
                                let err_msg = e.to_string();

                                // log
                                error!(target: "rag_query_handler", "{}", &err_msg);

                                return error::internal_server_error(err_msg);
                            }
                        };

                        // insert rag context into chat request
                        if let Err(e) = RagPromptBuilder::build(
                            &mut chat_request.messages,
                            &[context],
                            prompt_template.has_system_prompt(),
                            server_info.rag_config.policy,
                        ) {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "rag_query_handler", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    }
                }
            }
            None => {
                // log
                warn!(target: "rag_query_handler", "{}", format!("No point retrieved (score < threshold {})", server_info.qdrant_config.score_threshold
                ));
            }
        }
    }

    // chat completion
    let res = match llama_core::chat::chat(&mut chat_request).await {
        Ok(result) => match result {
            either::Left(stream) => {
                let stream = stream.map_err(|e| e.to_string());

                let result = Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "*")
                    .header("Access-Control-Allow-Headers", "*")
                    .header("Content-Type", "text/event-stream")
                    .header("Cache-Control", "no-cache")
                    .header("Connection", "keep-alive")
                    .header("user", id)
                    .body(Body::wrap_stream(stream));

                match result {
                    Ok(response) => {
                        // log
                        info!(target: "chat_completions_stream", "finish chat completions in stream mode");

                        response
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed chat completions in stream mode. Reason: {}", e);

                        // log
                        error!(target: "chat_completions_stream", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
            either::Right(chat_completion_object) => {
                // serialize chat completion object
                let s = match serde_json::to_string(&chat_completion_object) {
                    Ok(s) => s,
                    Err(e) => {
                        let err_msg = format!("Failed to serialize chat completion object. {}", e);

                        // log
                        error!(target: "chat_completions", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // return response
                let result = Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "*")
                    .header("Access-Control-Allow-Headers", "*")
                    .header("Content-Type", "application/json")
                    .header("user", id)
                    .body(Body::from(s));

                match result {
                    Ok(response) => {
                        // log
                        info!(target: "chat_completions", "Finish chat completions in non-stream mode");

                        response
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed chat completions in non-stream mode. Reason: {}", e);

                        // log
                        error!(target: "chat_completions", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
        },
        Err(e) => {
            let err_msg = format!("Failed to get chat completions. Reason: {}", e);

            // log
            error!(target: "chat_completions_handler", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    // log
    info!(target: "rag_query_handler", "Send the rag query response");

    res
}

#[derive(Debug, Default)]
struct RagPromptBuilder;
impl MergeRagContext for RagPromptBuilder {
    fn build(
        messages: &mut Vec<endpoints::chat::ChatCompletionRequestMessage>,
        context: &[String],
        has_system_prompt: bool,
        policy: MergeRagContextPolicy,
    ) -> ChatPromptsError::Result<()> {
        if messages.is_empty() {
            error!(target: "rag_prompt_builder", "No message in the chat request.");

            return Err(ChatPromptsError::PromptError::NoMessages);
        }

        if context.is_empty() {
            let err_msg = "No context provided.";

            // log
            error!(target: "rag_prompt_builder", "{}", &err_msg);

            return Err(ChatPromptsError::PromptError::Operation(err_msg.into()));
        }

        if policy == MergeRagContextPolicy::SystemMessage && !has_system_prompt {
            let err_msg = "The chat model does not support system message, while the given rag policy by '--policy' option requires that the RAG context is merged into system message. Please check the relevant CLI options and try again.";

            // log
            error!(target: "rag_prompt_builder", "{}", &err_msg);

            return Err(ChatPromptsError::PromptError::Operation(err_msg.into()));
        }

        info!(target: "rag_prompt_builder", "rag_policy: {}", &policy);

        let context = context[0].trim_end();

        info!(target: "rag_prompt_builder", "context:\n{}", context);

        match policy {
            MergeRagContextPolicy::SystemMessage => {
                info!(target: "rag_prompt_builder", "Merge RAG context into system message.");

                match &messages[0] {
                    ChatCompletionRequestMessage::System(message) => {
                        let system_message = {
                            // compose new system message content
                            let content = format!(
                                "{system_message}\n{context}",
                                system_message = message.content().trim(),
                                context = context.trim_end()
                            );
                            // create system message
                            ChatCompletionRequestMessage::new_system_message(
                                content,
                                message.name().cloned(),
                            )
                        };

                        // replace the original system message
                        messages[0] = system_message;
                    }
                    _ => {
                        let system_message = match GLOBAL_RAG_PROMPT.get() {
                            Some(global_rag_prompt) => {
                                // compose new system message content
                                let content = format!(
                                    "{rag_prompt}\n{context}",
                                    rag_prompt = global_rag_prompt.to_owned(),
                                    context = context.trim_end()
                                );
                                // create system message
                                ChatCompletionRequestMessage::new_system_message(content, None)
                            }
                            None => {
                                // compose new system message content
                                let content = format!("Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{}", context.trim_end());
                                // create system message
                                ChatCompletionRequestMessage::new_system_message(content, None)
                            }
                        };

                        // insert system message
                        messages.insert(0, system_message);
                    }
                }
            }
            MergeRagContextPolicy::LastUserMessage => {
                info!(target: "rag_prompt_builder", "Merge RAG context into last user message.");

                let len = messages.len();
                match &messages.last() {
                    Some(ChatCompletionRequestMessage::User(message)) => {
                        if let ChatCompletionUserMessageContent::Text(content) = message.content() {
                            // compose new user message content
                            let content = format!(
                                    "{context}\nAnswer the question based on the pieces of context above. The question is:\n{user_message}",
                                    context = context.trim_end(),
                                    user_message = content.trim(),
                                );

                            let content = ChatCompletionUserMessageContent::Text(content);

                            // create user message
                            let user_message = ChatCompletionRequestMessage::new_user_message(
                                content,
                                message.name().cloned(),
                            );
                            // replace the original user message
                            messages[len - 1] = user_message;
                        }
                    }
                    _ => {
                        let err_msg =
                            "The last message in the chat request should be a user message.";

                        // log
                        error!(target: "rag_prompt_builder", "{}", &err_msg);

                        return Err(ChatPromptsError::PromptError::BadMessages(err_msg.into()));
                    }
                }
            }
        }

        Ok(())
    }
}

pub(crate) async fn files_handler(req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "files_handler", "Handling the coming files request");

    let res = if req.method() == Method::POST {
        let boundary = "boundary=";

        let boundary = req.headers().get("content-type").and_then(|ct| {
            let ct = ct.to_str().ok()?;
            let idx = ct.find(boundary)?;
            Some(ct[idx + boundary.len()..].to_string())
        });

        let req_body = req.into_body();
        let body_bytes = match to_bytes(req_body).await {
            Ok(body_bytes) => body_bytes,
            Err(e) => {
                let err_msg = format!("Fail to read buffer from request body. {}", e);

                // log
                error!(target: "files_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        let cursor = Cursor::new(body_bytes.to_vec());

        let mut multipart = Multipart::with_body(cursor, boundary.unwrap());

        let mut file_object: Option<FileObject> = None;
        while let ReadEntryResult::Entry(mut field) = multipart.read_entry_mut() {
            if &*field.headers.name == "file" {
                let filename = match field.headers.filename {
                    Some(filename) => filename,
                    None => {
                        let err_msg =
                            "Failed to upload the target file. The filename is not provided.";

                        // log
                        error!(target: "files_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                if !((filename).to_lowercase().ends_with(".txt")
                    || (filename).to_lowercase().ends_with(".md"))
                    || (filename).to_lowercase().ends_with(".png")
                {
                    let err_msg = format!(
                        "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported. The file extension is {}.",
                        &filename
                    );

                    // log
                    error!(target: "files_handler", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                let mut buffer = Vec::new();
                let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                    Ok(size_in_bytes) => size_in_bytes,
                    Err(e) => {
                        let err_msg = format!("Failed to read the target file. {}", e);

                        // log
                        error!(target: "files_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // create a unique file id
                let id = format!("file_{}", uuid::Uuid::new_v4());

                // save the file
                let path = Path::new("archives");
                if !path.exists() {
                    fs::create_dir(path).unwrap();
                }
                let file_path = path.join(&id);
                if !file_path.exists() {
                    fs::create_dir(&file_path).unwrap();
                }
                let mut file = match File::create(file_path.join(&filename)) {
                    Ok(file) => file,
                    Err(e) => {
                        let err_msg =
                            format!("Failed to create archive document {}. {}", &filename, e);

                        // log
                        error!(target: "files_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };
                file.write_all(&buffer[..]).unwrap();

                // log
                info!(target: "files_handler", "file_id: {}, file_name: {}", &id, &filename);

                let created_at = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        error!(target: "files_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // create a file object
                file_object = Some(FileObject {
                    id,
                    bytes: size_in_bytes as u64,
                    created_at,
                    filename,
                    object: "file".to_string(),
                    purpose: "assistants".to_string(),
                });

                break;
            }
        }

        match file_object {
            Some(fo) => {
                // serialize chat completion object
                let s = match serde_json::to_string(&fo) {
                    Ok(s) => s,
                    Err(e) => {
                        let err_msg = format!("Failed to serialize file object. {}", e);

                        // log
                        error!(target: "files_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // return response
                let result = Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "*")
                    .header("Access-Control-Allow-Headers", "*")
                    .header("Content-Type", "application/json")
                    .body(Body::from(s));

                match result {
                    Ok(response) => response,
                    Err(e) => {
                        let err_msg = e.to_string();

                        // log
                        error!(target: "files_handler", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
            None => {
                let err_msg = "Failed to upload the target file. Not found the target file.";

                // log
                error!(target: "files_handler", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::GET {
        let uri_path = req.uri().path();

        if uri_path == "/v1/files" {
            let mut file_objects: Vec<FileObject> = Vec::new();
            for entry in WalkDir::new("archives").into_iter().filter_map(|e| e.ok()) {
                if !is_hidden(&entry) && entry.path().is_file() {
                    info!(target: "files_handler", "archive file: {}", entry.path().display());

                    let id = entry
                        .path()
                        .parent()
                        .and_then(|p| p.file_name())
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string();

                    let filename = entry
                        .path()
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap()
                        .to_string();

                    let metadata = entry.path().metadata().unwrap();

                    let created_at = metadata
                        .created()
                        .unwrap()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    let bytes = metadata.len();

                    let fo = FileObject {
                        id,
                        bytes,
                        created_at,
                        filename,
                        object: "file".to_string(),
                        purpose: "assistants".to_string(),
                    };

                    file_objects.push(fo);
                }
            }

            info!(target: "files_handler", "Found {} archive files", file_objects.len());

            let file_objects = ListFilesResponse {
                object: "list".to_string(),
                data: file_objects,
            };

            // serialize chat completion object
            let s = match serde_json::to_string(&file_objects) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Failed to serialize file object. {}", e);

                    // log
                    error!(target: "files_handler", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }
            };

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", "application/json")
                .body(Body::from(s));

            match result {
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "files_handler", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        } else {
            let id = uri_path.trim_start_matches("/v1/files/");
            let root = format!("archives/{}", id);
            let mut file_object: Option<FileObject> = None;
            for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
                if !is_hidden(&entry) && entry.path().is_file() {
                    info!(target: "files_handler", "archive file: {}", entry.path().display());

                    let filename = entry
                        .path()
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap()
                        .to_string();

                    let metadata = entry.path().metadata().unwrap();

                    let created_at = metadata
                        .created()
                        .unwrap()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    let bytes = metadata.len();

                    file_object = Some(FileObject {
                        id: id.into(),
                        bytes,
                        created_at,
                        filename,
                        object: "file".to_string(),
                        purpose: "assistants".to_string(),
                    });

                    break;
                }
            }

            match file_object {
                Some(fo) => {
                    // serialize chat completion object
                    let s = match serde_json::to_string(&fo) {
                        Ok(s) => s,
                        Err(e) => {
                            let err_msg = format!("Failed to serialize file object. {}", e);

                            // log
                            error!(target: "files_handler", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    };

                    // return response
                    let result = Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .header("Content-Type", "application/json")
                        .body(Body::from(s));

                    match result {
                        Ok(response) => response,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "files_handler", "{}", &err_msg);

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                None => {
                    let err_msg = format!(
                        "Failed to retrieve the target file. Not found the target file with id {}.",
                        id
                    );

                    // log
                    error!(target: "files_handler", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
    } else if req.method() == Method::DELETE {
        let id = req.uri().path().trim_start_matches("/v1/files/");
        let root = format!("archives/{}", id);
        let status = match fs::remove_dir_all(root) {
            Ok(_) => {
                info!(target: "files_handler", "Successfully deleted the target file with id {}.", id);

                DeleteFileStatus {
                    id: id.into(),
                    object: "file".to_string(),
                    deleted: true,
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to delete the target file with id {}. {}", id, e);

                // log
                error!(target: "files_handler", "{}", &err_msg);

                DeleteFileStatus {
                    id: id.into(),
                    object: "file".to_string(),
                    deleted: false,
                }
            }
        };

        // serialize status
        let s = match serde_json::to_string(&status) {
            Ok(s) => s,
            Err(e) => {
                let err_msg = format!(
                    "Failed to serialize the status of the file deletion operation. {}",
                    e
                );

                // log
                error!(target: "files_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        // return response
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::from(s));

        match result {
            Ok(response) => response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "files_handler", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else {
        let err_msg = "Invalid HTTP Method.";

        // log
        error!(target: "files_handler", "{}", &err_msg);

        error::internal_server_error(err_msg)
    };

    info!(target: "files_handler", "Send the files response");

    res
}

pub(crate) async fn chunks_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "chunks_handler", "Handling the coming chunks request");

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "chunks_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    let chunks_request: ChunksRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chunks_request) => chunks_request,
        Err(e) => {
            let err_msg = format!("Fail to deserialize chunks request: {msg}", msg = e);

            // log
            error!(target: "chunks_handler", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    // check if the archives directory exists
    let path = Path::new("archives");
    if !path.exists() {
        let err_msg = "The `archives` directory does not exist.";

        // log
        error!(target: "chunks_handler", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    // check if the archive id exists
    let archive_path = path.join(&chunks_request.id);
    if !archive_path.exists() {
        let err_msg = format!("Not found archive id: {}", &chunks_request.id);

        // log
        error!(target: "chunks_handler", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    // check if the file exists
    let file_path = archive_path.join(&chunks_request.filename);
    if !file_path.exists() {
        let err_msg = format!(
            "Not found file: {} in archive id: {}",
            &chunks_request.filename, &chunks_request.id
        );

        // log
        error!(target: "chunks_handler", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    // log
    info!(target: "chunks_handler", "file_id: {}, file_name: {}", &chunks_request.id, &chunks_request.filename);

    // get the extension of the archived file
    let extension = match file_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) => extension,
        None => {
            let err_msg = format!(
                "Failed to get the extension of the archived `{}`.",
                &chunks_request.filename
            );

            // log
            error!(target: "chunks_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // open the file
    let mut file = match File::open(&file_path) {
        Ok(file) => file,
        Err(e) => {
            let err_msg = format!("Failed to open `{}`. {}", &chunks_request.filename, e);

            // log
            error!(target: "chunks_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // read the file
    let mut contents = String::new();
    if let Err(e) = file.read_to_string(&mut contents) {
        let err_msg = format!("Failed to read `{}`. {}", &chunks_request.filename, e);

        // log
        error!(target: "chunks_handler", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    let res = match llama_core::rag::chunk_text(&contents, extension, chunks_request.chunk_capacity)
    {
        Ok(chunks) => {
            let chunks_response = ChunksResponse {
                id: chunks_request.id,
                filename: chunks_request.filename,
                chunks,
            };

            // serialize embedding object
            match serde_json::to_string(&chunks_response) {
                Ok(s) => {
                    // return response
                    let result = Response::builder()
                        .header("Access-Control-Allow-Origin", "*")
                        .header("Access-Control-Allow-Methods", "*")
                        .header("Access-Control-Allow-Headers", "*")
                        .header("Content-Type", "application/json")
                        .body(Body::from(s));
                    match result {
                        Ok(response) => response,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "chunks_handler", "{}", &err_msg);

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "chunks_handler", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("Fail to serialize chunks response. {}", e);

            // log
            error!(target: "chunks_handler", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "chunks_handler", "Send the chunks response.");

    res
}

pub(crate) async fn doc_to_embeddings_handler(
    req: Request<Body>,
    chunk_capacity: usize,
) -> Response<Body> {
    // log
    info!(target: "doc_to_embeddings_handler", "Handling the coming doc_to_embeddings request.");

    // upload the target rag document
    let file_object = if req.method() == Method::POST {
        let boundary = "boundary=";

        let boundary = req.headers().get("content-type").and_then(|ct| {
            let ct = ct.to_str().ok()?;
            let idx = ct.find(boundary)?;
            Some(ct[idx + boundary.len()..].to_string())
        });

        let req_body = req.into_body();
        let body_bytes = match to_bytes(req_body).await {
            Ok(body_bytes) => body_bytes,
            Err(e) => {
                let err_msg = format!("Fail to read buffer from request body. {}", e);

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        let cursor = Cursor::new(body_bytes.to_vec());

        let mut multipart = Multipart::with_body(cursor, boundary.unwrap());

        let mut file_object: Option<FileObject> = None;
        while let ReadEntryResult::Entry(mut field) = multipart.read_entry_mut() {
            if &*field.headers.name == "file" {
                let filename = match field.headers.filename {
                    Some(filename) => filename,
                    None => {
                        let err_msg =
                            "Failed to upload the target file. The filename is not provided.";

                        // log
                        error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                if !((filename).to_lowercase().ends_with(".txt")
                    || (filename).to_lowercase().ends_with(".md"))
                {
                    let err_msg = "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported.";

                    // log
                    error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                let mut buffer = Vec::new();
                let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                    Ok(size_in_bytes) => size_in_bytes,
                    Err(e) => {
                        let err_msg = format!("Failed to read the target file. {}", e);

                        // log
                        error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // create a unique file id
                let id = format!("file_{}", uuid::Uuid::new_v4());

                // save the file
                let path = Path::new("archives");
                if !path.exists() {
                    fs::create_dir(path).unwrap();
                }
                let file_path = path.join(&id);
                if !file_path.exists() {
                    fs::create_dir(&file_path).unwrap();
                }
                let mut file = match File::create(file_path.join(&filename)) {
                    Ok(file) => file,
                    Err(e) => {
                        let err_msg =
                            format!("Failed to create archive document {}. {}", &filename, e);

                        // log
                        error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };
                file.write_all(&buffer[..]).unwrap();

                // log
                info!(target: "doc_to_embeddings_handler", "file_id: {}, file_name: {}", &id, &filename);

                let created_at = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // create a file object
                file_object = Some(FileObject {
                    id,
                    bytes: size_in_bytes as u64,
                    created_at,
                    filename,
                    object: "file".to_string(),
                    purpose: "assistants".to_string(),
                });

                break;
            }
        }

        match file_object {
            Some(fo) => fo,
            None => {
                let err_msg = "Failed to upload the target file. Not found the target file.";

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    } else if req.method() == Method::GET {
        let err_msg = "Not implemented for listing files.";

        // log
        error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    } else {
        let err_msg = "Invalid HTTP Method.";

        // log
        error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    };

    // chunk the text
    let chunks = {
        info!(target: "doc_to_embeddings_handler", "file_id: {}, file_name: {}", &file_object.id, &file_object.filename);

        // check if the archives directory exists
        let path = Path::new("archives");
        if !path.exists() {
            let err_msg = "The `archives` directory does not exist.";

            // log
            error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }

        // check if the archive id exists
        let archive_path = path.join(&file_object.id);
        if !archive_path.exists() {
            let err_msg = format!("Not found archive id: {}", &file_object.id);

            // log
            error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }

        // check if the file exists
        let file_path = archive_path.join(&file_object.filename);
        if !file_path.exists() {
            let err_msg = format!(
                "Not found file: {} in archive id: {}",
                &file_object.filename, &file_object.id
            );

            // log
            error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }

        // get the extension of the archived file
        let extension = match file_path.extension().and_then(std::ffi::OsStr::to_str) {
            Some(extension) => extension,
            None => {
                let err_msg = format!(
                    "Failed to get the extension of the archived `{}`.",
                    &file_object.filename
                );

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        info!(target: "doc_to_embeddings_handler", "Open and read the file.");

        // open the file
        let mut file = match File::open(&file_path) {
            Ok(file) => file,
            Err(e) => {
                let err_msg = format!("Failed to open `{}`. {}", &file_object.filename, e);

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        // read the file
        let mut contents = String::new();
        if let Err(e) = file.read_to_string(&mut contents) {
            let err_msg = format!("Failed to read `{}`. {}", &file_object.filename, e);

            // log
            error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }

        info!(target: "doc_to_embeddings_handler", "Chunk the file contents.");

        match llama_core::rag::chunk_text(&contents, extension, chunk_capacity) {
            Ok(chunks) => chunks,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    };

    // compute embeddings for chunks
    let embedding_response = {
        // get the name of embedding model
        let model = match llama_core::utils::embedding_model_names() {
            Ok(model_names) => model_names[0].clone(),
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        info!(target: "doc_to_embeddings_handler", "Prepare the rag embedding request.");

        // create an embedding request
        let embedding_request = EmbeddingRequest {
            model,
            input: chunks.into(),
            encoding_format: None,
            user: None,
        };

        let server_info = match SERVER_INFO.get() {
            Some(server_info) => server_info,
            None => {
                let err_msg = "The server info is not set.";

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        // create rag embedding request
        let rag_embedding_request = RagEmbeddingRequest::from_embedding_request(
            embedding_request,
            server_info.qdrant_config.url.clone(),
            server_info.qdrant_config.collection_name.clone(),
        );

        match llama_core::rag::rag_doc_chunks_to_embeddings(&rag_embedding_request).await {
            Ok(embedding_response) => embedding_response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    };

    // serialize embedding response
    let res = match serde_json::to_string(&embedding_response) {
        Ok(s) => {
            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", "application/json")
                .body(Body::from(s));
            match result {
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("Fail to serialize embedding object. {}", e);

            // log
            error!(target: "doc_to_embeddings_handler", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "doc_to_embeddings_handler", "Send the doc_to_embeddings response.");

    res
}

pub(crate) async fn server_info_handler() -> Response<Body> {
    // log
    info!(target: "server_info", "Handling the coming server info request.");

    // get the server info
    let server_info = match SERVER_INFO.get() {
        Some(server_info) => server_info,
        None => {
            let err_msg = "The server info is not set.";

            // log
            error!(target: "server_info_handler", "{}", &err_msg);

            return error::internal_server_error("The server info is not set.");
        }
    };

    // serialize server info
    let s = match serde_json::to_string(&server_info) {
        Ok(s) => s,
        Err(e) => {
            let err_msg = format!("Fail to serialize server info. {}", e);

            // log
            error!(target: "server_info_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // return response
    let result = Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .header("Content-Type", "application/json")
        .body(Body::from(s));
    let res = match result {
        Ok(response) => response,
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "server_info_handler", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "server_info", "Send the server info response.");

    res
}

pub(crate) async fn retrieve_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "retrieve_handler", "Handling the coming retrieve request.");

    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body(Body::empty());

        match result {
            Ok(response) => return response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "retrieve_handler", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    info!(target: "rag_query_handler", "Prepare the chat completion request.");

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "retrieve_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            let err_msg = format!(
                "Fail to deserialize chat completion request: {msg}",
                msg = e
            );

            // log
            error!(target: "retrieve_handler", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    if chat_request.user.is_none() {
        chat_request.user = Some(gen_chat_id())
    };
    let id = chat_request.user.clone().unwrap();

    // log user id
    info!(target: "retrieve_handler", "user: {}", &id);

    let server_info = match SERVER_INFO.get() {
        Some(server_info) => server_info,
        None => {
            let err_msg = "The server info is not set.";

            // log
            error!(target: "retrieve_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    info!(target: "rag_query_handler", "Compute embeddings for user query.");

    // * compute embeddings for user query
    let embedding_response = match chat_request.messages.is_empty() {
        true => {
            let err_msg = "Messages should not be empty.";

            // log
            error!(target: "retrieve_handler", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
        false => {
            let last_message = chat_request.messages.last().unwrap();
            match last_message {
                ChatCompletionRequestMessage::User(user_message) => {
                    let query_text = match user_message.content() {
                        ChatCompletionUserMessageContent::Text(text) => text,
                        _ => {
                            let err_msg = "The last message must be a text content user message";

                            // log
                            error!(target: "retrieve_handler", "{}", &err_msg);

                            return error::bad_request(err_msg);
                        }
                    };

                    // log
                    info!(target: "retrieve_handler", "query_text: {}", query_text);

                    // get the available embedding models
                    let embedding_model_names = match llama_core::utils::embedding_model_names() {
                        Ok(model_names) => model_names,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "retrieve_handler", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    };

                    // create a embedding request
                    let embedding_request = EmbeddingRequest {
                        model: embedding_model_names[0].clone(),
                        input: query_text.into(),
                        encoding_format: None,
                        user: chat_request.user.clone(),
                    };

                    let rag_embedding_request = RagEmbeddingRequest {
                        embedding_request,
                        qdrant_url: server_info.qdrant_config.url.clone(),
                        qdrant_collection_name: server_info.qdrant_config.collection_name.clone(),
                    };

                    // compute embeddings for query
                    match llama_core::rag::rag_query_to_embeddings(&rag_embedding_request).await {
                        Ok(embedding_response) => embedding_response,
                        Err(e) => {
                            let err_msg = e.to_string();

                            // log
                            error!(target: "retrieve_handler", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    }
                }
                _ => {
                    let err_msg = "The last message must be a user message";

                    // log
                    error!(target: "retrieve_handler", "{}", &err_msg);

                    return error::bad_request(err_msg);
                }
            }
        }
    };
    let query_embedding: Vec<f32> = match embedding_response.data.first() {
        Some(embedding) => embedding.embedding.iter().map(|x| *x as f32).collect(),
        None => {
            let err_msg = "No embeddings returned";

            // log
            error!(target: "retrieve_handler", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // * retrieve context
    let res = match llama_core::rag::rag_retrieve_context(
        query_embedding.as_slice(),
        server_info.qdrant_config.url.to_string().as_str(),
        server_info.qdrant_config.collection_name.as_str(),
        server_info.qdrant_config.limit as usize,
        Some(server_info.qdrant_config.score_threshold),
    )
    .await
    {
        Ok(ro) => {
            // serialize retrieve object
            let s = match serde_json::to_string(&ro) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Fail to serialize retrieve object. {}", e);

                    // log
                    error!(target: "retrieve_handler", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }
            };

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", "application/json")
                .header("user", id)
                .body(Body::from(s));

            match result {
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "retrieve_handler", "{}", &err_msg);

                    error::internal_server_error(e.to_string())
                }
            }
        }
        Err(e) => {
            // log
            error!(target: "retrieve_handler", "{}", e);

            let ro = RetrieveObject {
                points: None,
                limit: server_info.qdrant_config.limit as usize,
                score_threshold: server_info.qdrant_config.score_threshold,
            };

            // serialize retrieve object
            let s = match serde_json::to_string(&ro) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Fail to serialize retrieve object. {}", e);

                    // log
                    error!(target: "retrieve_handler", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }
            };

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", "application/json")
                .header("user", id)
                .body(Body::from(s));

            match result {
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "retrieve_handler", "{}", &err_msg);

                    error::internal_server_error(e.to_string())
                }
            }
        }
    };

    info!(target: "retrieve_handler", "Send the retrieve response.");

    res
}

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with("."))
        .unwrap_or(false)
}
