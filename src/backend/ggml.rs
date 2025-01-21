use crate::{
    error, utils::gen_chat_id, QdrantConfig, CONTEXT_WINDOW, GLOBAL_RAG_PROMPT, KW_SEARCH_CONFIG,
    SERVER_INFO,
};
use chat_prompts::{error as ChatPromptsError, MergeRagContext, MergeRagContextPolicy};
use endpoints::{
    chat::{ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionUserMessageContent},
    embeddings::{ChunksRequest, ChunksResponse, EmbeddingRequest, InputText},
    files::{DeleteFileStatus, FileObject},
    keyword_search::{DocumentInput, IndexRequest, IndexResponse, QueryRequest, QueryResponse},
    rag::{CreateRagResponse, RagScoredPoint, RetrieveObject},
};
use futures_util::TryStreamExt;
use hyper::{body::to_bytes, Body, Method, Request, Response};
use llama_core::{
    embeddings::{chunk_text, embeddings},
    rag::{rag_doc_chunks_to_embeddings, rag_query_to_embeddings, rag_retrieve_context},
};
use multipart::server::{Multipart, ReadEntry, ReadEntryResult};
use multipart_2021 as multipart;
use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{Cursor, Read, Write},
    path::Path,
    time::SystemTime,
};

/// List all models available.
pub(crate) async fn models_handler() -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming model list request.");

    let list_models_response = match llama_core::models::models().await {
        Ok(list_models_response) => list_models_response,
        Err(e) => {
            let err_msg = format!("Failed to get model list. Reason: {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // serialize response
    let s = match serde_json::to_string(&list_models_response) {
        Ok(s) => s,
        Err(e) => {
            let err_msg = format!("Failed to serialize the model list result. Reason: {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

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
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    // log
    info!(target: "stdout", "Send the model list response.");

    res
}

/// Compute embeddings for the input text and return the embeddings object.
pub(crate) async fn embeddings_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming embeddings request");

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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut embedding_request: EmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(embedding_request) => embedding_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize embedding request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    if embedding_request.user.is_none() {
        embedding_request.user = Some(gen_chat_id())
    };
    let id = embedding_request.user.clone().unwrap();

    // log user id
    info!(target: "stdout", "user: {}", &id);

    let res = match embeddings(&embedding_request).await {
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
                            error!(target: "stdout", "{}", &err_msg);

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Fail to serialize embedding object. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = e.to_string();

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the embeddings response");

    res
}

/// Query a user input and return a chat-completion response with the answer from the model.
///
/// Note that the body of the request is deserialized to a `ChatCompletionRequest` instance.
pub(crate) async fn rag_query_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming rag query request");

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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    info!(target: "stdout", "Prepare the chat completion request");

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize chat completion request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            // log body_bytes
            error!(target: "stdout", "raw data:\n{:?}", &body_bytes.to_ascii_lowercase());

            return error::bad_request(err_msg);
        }
    };

    // check if the user id is provided
    if chat_request.user.is_none() {
        chat_request.user = Some(gen_chat_id())
    };
    let id = chat_request.user.clone().unwrap();

    // log user id
    info!(target: "stdout", "user: {}", &id);

    // perform keyword search
    let mut kw_hits = Vec::new();
    let mut kw_search_url = match &chat_request.kw_search_url {
        Some(url) => url.clone(),
        None => match KW_SEARCH_CONFIG.get() {
            Some(kw_search_config) => kw_search_config.url.clone(),
            None => String::new(),
        },
    };
    if !kw_search_url.is_empty() {
        kw_search_url = kw_search_url.trim_end_matches('/').to_string();
        info!(target: "stdout", "kw_search_url: {}", &kw_search_url);

        if let Some(index_name) = &chat_request.kw_index_name {
            if !index_name.is_empty() {
                if let Some(ChatCompletionRequestMessage::User(user_message)) =
                    chat_request.messages.last()
                {
                    if let ChatCompletionUserMessageContent::Text(text) = user_message.content() {
                        info!(target: "stdout", "perform keyword search on the index: {}", &index_name);

                        let kw_top_k = chat_request.kw_top_k.unwrap();

                        let user_query = text.clone();
                        let query_request = QueryRequest {
                            query: user_query,
                            top_k: kw_top_k as usize,
                            index: index_name.clone(),
                        };

                        let query_url = format!("{}/v1/search", &kw_search_url);
                        info!(target: "stdout", "query_url: {}", &query_url);

                        // send query request to the keyword search service
                        match reqwest::Client::new()
                            .post(&query_url)
                            .json(&query_request)
                            .send()
                            .await
                        {
                            Ok(response) => {
                                match response.json::<QueryResponse>().await {
                                    Ok(query_response) => {
                                        match query_response.error {
                                            Some(error) => {
                                                let err_msg = format!(
                                                    "Failed to perform keyword search. Reason: {}",
                                                    error
                                                );

                                                // log
                                                warn!(target: "stdout", "{}", &err_msg);
                                            }
                                            None => {
                                                info!(target: "stdout", "Number of keyword search hits: {}", &query_response.hits.len());

                                                kw_hits = query_response.hits;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let err_msg = format!(
                                            "Failed to perform keyword search. Reason: {}",
                                            e
                                        );

                                        // log
                                        warn!(target: "stdout", "{}", &err_msg);
                                    }
                                }
                            }
                            Err(e) => {
                                let err_msg =
                                    format!("Failed to perform keyword search. Reason: {}", e);

                                // log
                                warn!(target: "stdout", "{}", &err_msg);
                            }
                        }
                    }
                }
            }
        }
    }

    // qdrant config
    let qdrant_config_vec = match get_qdrant_configs(&chat_request).await {
        Ok(qdrant_config_vec) => qdrant_config_vec,
        Err(e) => return error::internal_server_error(e.to_string()),
    };

    // retrieve context
    let mut retrieve_object_vec = match retrieve_context_with_multiple_qdrant_configs(
        &chat_request,
        &qdrant_config_vec,
    )
    .await
    {
        Ok(retrieve_object_vec) => retrieve_object_vec,
        Err(response) => {
            return response;
        }
    };

    // log retrieve object
    debug!(target: "stdout", "retrieve_object_vec:\n{}", serde_json::to_string_pretty(&retrieve_object_vec).unwrap());

    // fuse kw-search and embedding-search results
    if !kw_hits.is_empty()
        && !retrieve_object_vec.is_empty()
        && retrieve_object_vec[0].points.is_some()
    {
        let points = retrieve_object_vec[0].points.as_ref().unwrap().clone();
        if !points.is_empty() {
            let limit = retrieve_object_vec[0].limit;
            let score_threshold = retrieve_object_vec[0].score_threshold;

            // create a hash map from retrieve_object_vec: key is the hash value of the source of the point, value is the point
            let mut em_hits_map = HashMap::new();
            let mut em_scores = HashMap::new();

            for point in points {
                let hash_value = calculate_hash(&point.source);
                em_scores.insert(hash_value, point.score);
                em_hits_map.insert(hash_value, point);
            }

            info!(target: "stdout", "em_hits_map: {:#?}", &em_hits_map);

            // normalize the em_scores
            let em_scores = normalize(&em_scores);

            info!(target: "stdout", "em_scores: {:#?}", &em_scores);

            // create a hash map from kw_hits: key is the hash value of the content of the hit, value is the hit
            let mut kw_hits_map = HashMap::new();
            let mut kw_scores = HashMap::new();
            for hit in kw_hits {
                let hash_value = calculate_hash(&hit.content);
                kw_scores.insert(hash_value, hit.score);
                kw_hits_map.insert(hash_value, hit);
            }

            info!(target: "stdout", "kw_hits_map: {:#?}", &kw_hits_map);

            // normalize the kw_scores
            let kw_scores = normalize(&kw_scores);

            info!(target: "stdout", "kw_scores: {:#?}", &kw_scores);

            // Set weight alpha
            let alpha = 0.7;

            // fuse the two hash maps
            let final_scores = weighted_fusion(kw_scores, em_scores, alpha);

            info!(target: "stdout", "final_scores: {:#?}", &final_scores);

            // Sort by score from high to low
            let mut final_ranking: Vec<(u64, f32)> = final_scores.into_iter().collect();
            final_ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Print final ranking
            info!(target: "stdout", "final_ranking: {:#?}", &final_ranking);

            let mut retrieved = Vec::new();
            for (hash_value, score) in final_ranking {
                if score >= score_threshold {
                    let mut doc = RagScoredPoint {
                        source: String::new(),
                        score,
                    };
                    if kw_hits_map.contains_key(&hash_value) {
                        doc.source = kw_hits_map[&hash_value].content.clone();
                        retrieved.push(doc);
                    } else if em_hits_map.contains_key(&hash_value) {
                        doc.source = em_hits_map[&hash_value].source.clone();
                        retrieved.push(doc);
                    }
                }
            }

            if retrieved.len() > limit {
                retrieved.truncate(limit);
            }

            info!(target: "stdout", "retrieved: {:#?}", &retrieved);

            let retrieve_object = RetrieveObject {
                limit,
                score_threshold,
                points: Some(retrieved),
            };

            retrieve_object_vec = vec![retrieve_object];
        }
    }

    // * extract the context from retrieved objects
    let mut context = String::new();
    for (idx, retrieve_object) in retrieve_object_vec.iter().enumerate() {
        match retrieve_object.points.as_ref() {
            Some(scored_points) => {
                match scored_points.is_empty() {
                    false => {
                        for (idx, point) in scored_points.iter().enumerate() {
                            // log
                            info!(target: "stdout", "point: {}, score: {}, source: {}", idx, point.score, &point.source);

                            context.push_str(&point.source);
                            context.push_str("\n\n");
                        }
                    }
                    true => {
                        // log
                        warn!(target: "stdout", "{}", format!("No point retrieved from the collection `{}` (score < threshold {})", qdrant_config_vec[idx].collection_name, qdrant_config_vec[idx].score_threshold));
                    }
                }
            }
            None => {
                // log
                warn!(target: "stdout", "{}", format!("No point retrieved from the collection `{}` (score < threshold {})", qdrant_config_vec[idx].collection_name, qdrant_config_vec[idx].score_threshold));
            }
        }
    }

    // * update messages with retrieved context
    if !context.is_empty() {
        if chat_request.messages.is_empty() {
            let err_msg = "No message in the chat request.";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }

        let prompt_template =
            match llama_core::utils::chat_prompt_template(chat_request.model.as_deref()) {
                Ok(prompt_template) => prompt_template,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }
            };

        let rag_policy = match SERVER_INFO.get() {
            Some(server_info) => server_info.read().await.rag_config.policy,
            None => {
                let err_msg = "SERVER_INFO is not initialized.";

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        // insert rag context into chat request
        if let Err(e) = RagPromptBuilder::build(
            &mut chat_request.messages,
            &[context],
            prompt_template.has_system_prompt(),
            rag_policy,
        ) {
            let err_msg = e.to_string();

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    }

    // * perform chat completion
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
                        info!(target: "stdout", "finish chat completions in stream mode");

                        response
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed chat completions in stream mode. Reason: {}", e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

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
                        error!(target: "stdout", "{}", &err_msg);

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
                        info!(target: "stdout", "Finish chat completions in non-stream mode");

                        response
                    }
                    Err(e) => {
                        let err_msg =
                            format!("Failed chat completions in non-stream mode. Reason: {}", e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
        },
        Err(e) => {
            let err_msg = format!("Failed to get chat completions. Reason: {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    // log
    info!(target: "stdout", "Send the rag query response");

    res
}

async fn retrieve_context_with_single_qdrant_config(
    chat_request: &ChatCompletionRequest,
    qdrant_config: &QdrantConfig,
) -> Result<RetrieveObject, Response<Body>> {
    info!(target: "stdout", "Compute embeddings for user query.");

    // get context_window: chat_request.context_window prioritized CONTEXT_WINDOW
    let context_window = chat_request
        .context_window
        .or_else(|| CONTEXT_WINDOW.get().copied())
        .unwrap_or(1);
    info!(target: "stdout", "context window: {}", context_window);

    info!(target: "stdout", "VectorDB config: {}", qdrant_config);

    // compute embeddings for user query
    let embedding_response = match chat_request.messages.is_empty() {
        true => {
            let err_msg = "Messages should not be empty.";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return Err(error::bad_request(err_msg));
        }
        false => {
            // get the last `n` user messages in the context window.
            // `n` is determined by the `context_window` in the chat request.
            let mut last_n_user_messages = Vec::new();
            for (idx, message) in chat_request.messages.iter().rev().enumerate() {
                if let ChatCompletionRequestMessage::User(user_message) = message {
                    if let ChatCompletionUserMessageContent::Text(text) = user_message.content() {
                        if !text.ends_with("<server-health>") {
                            last_n_user_messages.push(text.clone());
                        } else if idx == 0 {
                            let content = text.trim_end_matches("<server-health>").to_string();
                            last_n_user_messages.push(content);
                            break;
                        }
                    }
                }

                if last_n_user_messages.len() == context_window as usize {
                    break;
                }
            }

            // join the user messages in the context window into a single string
            let query_text = if !last_n_user_messages.is_empty() {
                info!(target: "stdout", "Found the latest {} user messages", last_n_user_messages.len());

                last_n_user_messages.reverse();
                last_n_user_messages.join("\n")
            } else {
                let warn_msg = "No user messages found.";

                // log
                warn!(target: "stdout", "{}", &warn_msg);

                return Err(error::bad_request(warn_msg));
            };

            // log
            info!(target: "stdout", "query text for the context retrieval: {}", query_text);

            // get the available embedding models
            let embedding_model_names = match llama_core::utils::embedding_model_names() {
                Ok(model_names) => model_names,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(error::internal_server_error(err_msg));
                }
            };

            // get vdb_api_key if it is provided in the request, otherwise get it from the environment variable `VDB_API_KEY`
            let vdb_api_key = chat_request
                .vdb_api_key
                .clone()
                .or_else(|| std::env::var("VDB_API_KEY").ok());

            // create a embedding request
            let embedding_request = EmbeddingRequest {
                model: Some(embedding_model_names[0].clone()),
                input: InputText::String(query_text),
                encoding_format: None,
                user: chat_request.user.clone(),
                vdb_server_url: Some(qdrant_config.url.clone()),
                vdb_collection_name: Some(qdrant_config.collection_name.clone()),
                vdb_api_key,
            };

            // compute embeddings for query
            match rag_query_to_embeddings(&embedding_request).await {
                Ok(embedding_response) => embedding_response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return Err(error::internal_server_error(err_msg));
                }
            }
        }
    };
    let query_embedding: Vec<f32> = match embedding_response.data.first() {
        Some(embedding) => embedding.embedding.iter().map(|x| *x as f32).collect(),
        None => {
            let err_msg = "No embeddings returned";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return Err(error::internal_server_error(err_msg));
        }
    };

    // get vdb_api_key if it is provided in the request, otherwise get it from the environment variable `VDB_API_KEY`
    let vdb_api_key = chat_request
        .vdb_api_key
        .clone()
        .or_else(|| std::env::var("VDB_API_KEY").ok());

    // perform the context retrieval
    let mut retrieve_object: RetrieveObject = match rag_retrieve_context(
        query_embedding.as_slice(),
        qdrant_config.url.to_string().as_str(),
        qdrant_config.collection_name.as_str(),
        qdrant_config.limit as usize,
        Some(qdrant_config.score_threshold),
        vdb_api_key,
    )
    .await
    {
        Ok(search_result) => search_result,
        Err(e) => {
            let err_msg = format!("No point retrieved. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return Err(error::internal_server_error(err_msg));
        }
    };
    if retrieve_object.points.is_none() {
        retrieve_object.points = Some(Vec::new());
    }

    info!(target: "stdout", "{} point(s) retrieved from the collection `{}`", retrieve_object.points.as_ref().unwrap().len(), qdrant_config.collection_name);

    Ok(retrieve_object)
}

async fn retrieve_context_with_multiple_qdrant_configs(
    chat_request: &ChatCompletionRequest,
    qdrant_config_vec: &[QdrantConfig],
) -> Result<Vec<RetrieveObject>, Response<Body>> {
    let mut retrieve_object_vec: Vec<RetrieveObject> = Vec::new();
    let mut set: HashSet<String> = HashSet::new();
    for qdrant_config in qdrant_config_vec {
        let mut retrieve_object =
            retrieve_context_with_single_qdrant_config(chat_request, qdrant_config).await?;

        if let Some(points) = retrieve_object.points.as_mut() {
            if !points.is_empty() {
                // find the duplicate points
                let mut idx_removed = vec![];
                for (idx, point) in points.iter().enumerate() {
                    if set.contains(&point.source) {
                        idx_removed.push(idx);
                    } else {
                        set.insert(point.source.clone());
                    }
                }

                // remove the duplicate points
                if !idx_removed.is_empty() {
                    let num = idx_removed.len();

                    for idx in idx_removed.iter().rev() {
                        points.remove(*idx);
                    }

                    info!(target: "stdout", "removed duplicated {} point(s) retrieved from the collection `{}`", num, qdrant_config.collection_name);
                }

                if !points.is_empty() {
                    retrieve_object_vec.push(retrieve_object);
                }
            }
        }
    }

    Ok(retrieve_object_vec)
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
            error!(target: "stdout", "No message in the chat request.");

            return Err(ChatPromptsError::PromptError::NoMessages);
        }

        if context.is_empty() {
            let err_msg = "No context provided.";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return Err(ChatPromptsError::PromptError::Operation(err_msg.into()));
        }

        if policy == MergeRagContextPolicy::SystemMessage && !has_system_prompt {
            let err_msg = "The chat model does not support system message, while the given rag policy by '--policy' option requires that the RAG context is merged into system message. Please check the relevant CLI options and try again.";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return Err(ChatPromptsError::PromptError::Operation(err_msg.into()));
        }
        info!(target: "stdout", "rag_policy: {}", &policy);

        let context = context[0].trim_end();
        info!(target: "stdout", "context:\n{}", context);

        match policy {
            MergeRagContextPolicy::SystemMessage => {
                info!(target: "stdout", "Merge RAG context into system message.");

                match &messages[0] {
                    ChatCompletionRequestMessage::System(message) => {
                        let system_message = {
                            match GLOBAL_RAG_PROMPT.get() {
                                Some(global_rag_prompt) => {
                                    // compose new system message content
                                    let content = format!(
                                        "{system_message}\n{rag_prompt}\n{context}",
                                        system_message = message.content().trim(),
                                        rag_prompt = global_rag_prompt.to_owned(),
                                        context = context
                                    );

                                    // log
                                    info!(target: "stdout", "system message with RAG context: {}", &content);

                                    // create system message
                                    ChatCompletionRequestMessage::new_system_message(
                                        content,
                                        message.name().cloned(),
                                    )
                                }
                                None => {
                                    // compose new system message content
                                    let content = format!(
                                        "{system_message}\n{context}",
                                        system_message = message.content().trim(),
                                        context = context
                                    );

                                    // log
                                    info!(target: "stdout", "system message with RAG context: {}", &content);

                                    // create system message
                                    ChatCompletionRequestMessage::new_system_message(
                                        content,
                                        message.name().cloned(),
                                    )
                                }
                            }
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
                                    context = context
                                );

                                // log
                                info!(target: "stdout", "system message with RAG context: {}", &content);

                                // create system message
                                ChatCompletionRequestMessage::new_system_message(content, None)
                            }
                            None => {
                                // compose new system message content
                                let content = context.to_string();

                                // log
                                info!(target: "stdout", "system message with RAG context: {}", &content);

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
                info!(target: "stdout", "Merge RAG context into last user message.");

                let len = messages.len();
                match &messages.last() {
                    Some(ChatCompletionRequestMessage::User(message)) => {
                        if let ChatCompletionUserMessageContent::Text(content) = message.content() {
                            // compose new user message content
                            let content = format!(
                                    "{context}\nAnswer the question based on the pieces of context above. The question is:\n{user_message}",
                                    context = context,
                                    user_message = content.trim(),
                                );

                            // log
                            info!(target: "stdout", "last user message with RAG context: {}", &content);

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
                        error!(target: "stdout", "{}", &err_msg);

                        return Err(ChatPromptsError::PromptError::BadMessages(err_msg.into()));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Upload, download, retrieve and delete a file, or list all files.
///
/// - `POST /v1/files`: Upload a file.
/// - `GET /v1/files`: List all files.
/// - `GET /v1/files/{file_id}`: Retrieve a file by id.
/// - `GET /v1/files/{file_id}/content`: Retrieve the content of a file by id.
/// - `GET /v1/files/download/{file_id}`: Download a file by id.
/// - `DELETE /v1/files/{file_id}`: Delete a file by id.
///
pub(crate) async fn files_handler(req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming files request");

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
                error!(target: "stdout", "{}", &err_msg);

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
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                if !((filename).to_lowercase().ends_with(".txt")
                    || (filename).to_lowercase().ends_with(".md")
                    || (filename).to_lowercase().ends_with(".png")
                    || (filename).to_lowercase().ends_with(".wav"))
                {
                    let err_msg = format!(
                        "Failed to upload the target file. Only files with 'txt', 'md', 'png', 'wav' extensions are supported. The file to be uploaded is {}.",
                        &filename
                    );

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                let mut buffer = Vec::new();
                let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                    Ok(size_in_bytes) => size_in_bytes,
                    Err(e) => {
                        let err_msg = format!("Failed to read the target file. {}", e);

                        // log
                        error!(target: "stdout", "{}", &err_msg);

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
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };
                file.write_all(&buffer[..]).unwrap();

                // log
                info!(target: "stdout", "file_id: {}, file_name: {}", &id, &filename);

                let created_at = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

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
                        error!(target: "stdout", "{}", &err_msg);

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
                        error!(target: "stdout", "{}", &err_msg);

                        error::internal_server_error(err_msg)
                    }
                }
            }
            None => {
                let err_msg = "Failed to upload the target file. Not found the target file.";

                // log
                error!(target: "stdout", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::GET {
        let uri_path = req.uri().path().trim_end_matches('/').to_lowercase();

        // Split the path into segments
        let segments: Vec<&str> = uri_path.split('/').collect();

        match segments.as_slice() {
            ["", "v1", "files"] => list_files(),
            ["", "v1", "files", file_id, "content"] => {
                if !file_id.starts_with("file_") {
                    let err_msg = format!("unsupported uri path: {}", uri_path);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                retrieve_file_content(file_id)
            }
            ["", "v1", "files", file_id] => {
                if !file_id.starts_with("file_") {
                    let err_msg = format!("unsupported uri path: {}", uri_path);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }

                retrieve_file(file_id)
            }
            ["", "v1", "files", "download", file_id] => download_file(file_id),
            _ => {
                let err_msg = format!("unsupported uri path: {}", uri_path);

                // log
                error!(target: "stdout", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::DELETE {
        let id = req.uri().path().trim_start_matches("/v1/files/");
        let status = match llama_core::files::remove_file(id) {
            Ok(status) => status,
            Err(e) => {
                let err_msg = format!("Failed to delete the target file with id {}. {}", id, e);

                // log
                error!(target: "stdout", "{}", &err_msg);

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
                error!(target: "stdout", "{}", &err_msg);

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
                error!(target: "stdout", "{}", &err_msg);

                error::internal_server_error(err_msg)
            }
        }
    } else if req.method() == Method::OPTIONS {
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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    } else {
        let err_msg = "Invalid HTTP Method.";

        // log
        error!(target: "stdout", "{}", &err_msg);

        error::internal_server_error(err_msg)
    };

    info!(target: "stdout", "Send the files response");

    res
}

fn list_files() -> Response<Body> {
    match llama_core::files::list_files() {
        Ok(file_objects) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&file_objects) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Failed to serialize file list. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

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
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("Failed to list all files. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

fn retrieve_file(id: impl AsRef<str>) -> Response<Body> {
    match llama_core::files::retrieve_file(id) {
        Ok(fo) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&fo) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Failed to serialize file object. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

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
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("{}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

fn retrieve_file_content(id: impl AsRef<str>) -> Response<Body> {
    match llama_core::files::retrieve_file_content(id) {
        Ok(content) => {
            // serialize chat completion object
            let s = match serde_json::to_string(&content) {
                Ok(s) => s,
                Err(e) => {
                    let err_msg = format!("Failed to serialize file content. {}", e);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

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
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("{}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

fn download_file(id: impl AsRef<str>) -> Response<Body> {
    match llama_core::files::download_file(id) {
        Ok((filename, buffer)) => {
            // get the extension of the file
            let extension = filename.split('.').next_back().unwrap_or("unknown");
            let content_type = match extension {
                "txt" => "text/plain",
                "json" => "application/json",
                "png" => "image/png",
                "jpg" => "image/jpeg",
                "jpeg" => "image/jpeg",
                "wav" => "audio/wav",
                "mp3" => "audio/mpeg",
                "mp4" => "video/mp4",
                "md" => "text/markdown",
                _ => {
                    let err_msg = format!("Unsupported file extension: {}", extension);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }
            };
            let content_disposition = format!("attachment; filename={}", filename);

            // return response
            let result = Response::builder()
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "*")
                .header("Access-Control-Allow-Headers", "*")
                .header("Content-Type", content_type)
                .header("Content-Disposition", content_disposition)
                .body(Body::from(buffer));

            match result {
                Ok(response) => response,
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("{}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    }
}

pub(crate) async fn chunks_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming chunks request");

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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    let chunks_request: ChunksRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chunks_request) => chunks_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize chunks request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    // check if the archives directory exists
    let path = Path::new("archives");
    if !path.exists() {
        let err_msg = "The `archives` directory does not exist.";

        // log
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    // check if the archive id exists
    let archive_path = path.join(&chunks_request.id);
    if !archive_path.exists() {
        let err_msg = format!("Not found archive id: {}", &chunks_request.id);

        // log
        error!(target: "stdout", "{}", &err_msg);

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
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    // log
    info!(target: "stdout", "file_id: {}, file_name: {}", &chunks_request.id, &chunks_request.filename);

    // get the extension of the archived file
    let extension = match file_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) => extension,
        None => {
            let err_msg = format!(
                "Failed to get the extension of the archived `{}`.",
                &chunks_request.filename
            );

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // open the file
    let mut file = match File::open(&file_path) {
        Ok(file) => file,
        Err(e) => {
            let err_msg = format!("Failed to open `{}`. {}", &chunks_request.filename, e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };

    // read the file
    let mut contents = String::new();
    if let Err(e) = file.read_to_string(&mut contents) {
        let err_msg = format!("Failed to read `{}`. {}", &chunks_request.filename, e);

        // log
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    }

    let res = match chunk_text(&contents, extension, chunks_request.chunk_capacity) {
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
                            error!(target: "stdout", "{}", &err_msg);

                            error::internal_server_error(err_msg)
                        }
                    }
                }
                Err(e) => {
                    let err_msg = e.to_string();

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("Fail to serialize chunks response. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the chunks response.");

    res
}

pub(crate) async fn create_rag_handler(
    req: Request<Body>,
    chunk_capacity: usize,
) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming doc_to_embeddings request.");

    // upload the target rag document
    let (file_object, vdb_server_url, vdb_collection_name, vdb_api_key, kw_search_url) = if req
        .method()
        == Method::POST
    {
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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        let cursor = Cursor::new(body_bytes.to_vec());

        let mut multipart = Multipart::with_body(cursor, boundary.unwrap());

        let mut file_object: Option<FileObject> = None;
        let mut vdb_server_url: String = String::new();
        let mut vdb_collection_name: String = String::new();
        let mut vdb_api_key: String = String::new();
        let mut kw_search_url = String::new();
        while let ReadEntryResult::Entry(mut field) = multipart.read_entry_mut() {
            match &*field.headers.name {
                "file" => {
                    let filename = match field.headers.filename {
                        Some(filename) => filename,
                        None => {
                            let err_msg =
                                "Failed to upload the target file. The filename is not provided.";

                            // log
                            error!(target: "stdout", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    };

                    info!(target: "stdout", "filename: {}", &filename);

                    if !((filename).to_lowercase().ends_with(".txt")
                        || (filename).to_lowercase().ends_with(".md"))
                    {
                        let err_msg = "Failed to upload the target file. Only files with 'txt' and 'md' extensions are supported.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }

                    let mut buffer = Vec::new();
                    let size_in_bytes = match field.data.read_to_end(&mut buffer) {
                        Ok(size_in_bytes) => size_in_bytes,
                        Err(e) => {
                            let err_msg = format!("Failed to read the target file. {}", e);

                            // log
                            error!(target: "stdout", "{}", &err_msg);

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
                            error!(target: "stdout", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    };
                    file.write_all(&buffer[..]).unwrap();

                    // log
                    info!(target: "stdout", "file_id: {}, file_name: {}", &id, &filename);

                    let created_at = match SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                        Ok(n) => n.as_secs(),
                        Err(_) => {
                            let err_msg = "Failed to get the current time.";

                            // log
                            error!(target: "stdout", "{}", &err_msg);

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
                }
                "vdb_server_url" => match field.is_text() {
                    true => {
                        if let Err(e) = field.data.read_to_string(&mut vdb_server_url) {
                            let err_msg =
                                format!("Failed to read the `vdb_server_url` field. {}", e);

                            // log
                            error!(target: "stdout", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    }
                    false => {
                        let err_msg =
                        "Failed to get `vdb_server_url`. The `vdb_server_url` field in the request should be a text field.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                },
                "vdb_collection_name" => match field.is_text() {
                    true => {
                        if let Err(e) = field.data.read_to_string(&mut vdb_collection_name) {
                            let err_msg =
                                format!("Failed to read the `vdb_collection_name` field. {}", e);

                            // log
                            error!(target: "stdout", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    }
                    false => {
                        let err_msg = "Failed to get `vdb_collection_name`. The `vdb_collection_name` field in the request should be a text field.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                },
                "vdb_api_key" => match field.is_text() {
                    true => {
                        if let Err(e) = field.data.read_to_string(&mut vdb_api_key) {
                            let err_msg = format!("Failed to read the `vdb_api_key` field. {}", e);

                            // log
                            error!(target: "stdout", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    }
                    false => {
                        let err_msg = "Failed to get `vdb_api_key`. The `vdb_api_key` field in the request should be a text field.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                },
                "kw_search_url" => match field.is_text() {
                    true => {
                        if let Err(e) = field.data.read_to_string(&mut kw_search_url) {
                            let err_msg =
                                format!("Failed to read the `kw_search_url` field. {}", e);

                            // log
                            error!(target: "stdout", "{}", &err_msg);

                            return error::internal_server_error(err_msg);
                        }
                    }
                    false => {
                        let err_msg = "Failed to get `kw_search_url`. The `kw_search_url` field in the request should be a text field.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                },
                _ => {
                    let err_msg = format!("Invalid field name: {}", &field.headers.name);

                    // log
                    error!(target: "stdout", "{}", &err_msg);

                    return error::internal_server_error(err_msg);
                }
            }
        }

        // If the request does not provide the vdb_server_url and vdb_collection_name, use the default vdb config from the server info, and get the vdb_api_key from the environment variable `VDB_API_KEY` if it is set.
        match (vdb_server_url.is_empty(), vdb_collection_name.is_empty()) {
            (true, true) => {
                let qdrant_config_vec = match SERVER_INFO.get() {
                    Some(server_info) => server_info.read().await.qdrant_config.clone(),
                    None => {
                        let err_msg = "The server info is not set.";

                        // log
                        error!(target: "stdout", "{}", &err_msg);

                        return error::internal_server_error(err_msg);
                    }
                };

                // use the first qdrant config as the default config
                vdb_server_url = qdrant_config_vec[0].url.clone();
                vdb_collection_name = qdrant_config_vec[0].collection_name.clone();
                if vdb_api_key.is_empty() {
                    vdb_api_key = std::env::var("VDB_API_KEY").unwrap_or_default();
                }
            }
            (true, false) | (false, true) => {
                let err_msg = "Failed to get `vdb_server_url` or `vdb_collection_name`. The `vdb_server_url` and `vdb_collection_name` fields in the request should be provided at the same time.";

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
            (false, false) => {}
        }

        // if kw_search_url is not provided, try to get the default kw_search_url from KW_SEARCH_CONFIG
        if kw_search_url.is_empty() {
            if let Some(kw_search_config) = KW_SEARCH_CONFIG.get() {
                kw_search_url = kw_search_config.url.clone();
            }
        }
        if !kw_search_url.is_empty() {
            kw_search_url = kw_search_url.trim_end_matches('/').to_string();
            info!(target: "stdout", "kw_search_url: {}", &kw_search_url);
        }

        info!(target: "stdout", "vdb_server_url: {}, vdb_collection_name: {}", &vdb_server_url, &vdb_collection_name);

        match file_object {
            Some(fo) => (
                fo,
                vdb_server_url,
                vdb_collection_name,
                vdb_api_key,
                kw_search_url,
            ),
            None => {
                let err_msg = "Failed to upload the target file. Not found the target file.";

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    } else if req.method() == Method::GET {
        let err_msg = "Not implemented for listing files.";

        // log
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    } else if req.method() == Method::OPTIONS {
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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    } else {
        let err_msg = "Invalid HTTP Method.";

        // log
        error!(target: "stdout", "{}", &err_msg);

        return error::internal_server_error(err_msg);
    };

    // chunk the text
    let chunks = {
        info!(target: "stdout", "file_id: {}, file_name: {}", &file_object.id, &file_object.filename);

        // check if the archives directory exists
        let path = Path::new("archives");
        if !path.exists() {
            let err_msg = "The `archives` directory does not exist.";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }

        // check if the archive id exists
        let archive_path = path.join(&file_object.id);
        if !archive_path.exists() {
            let err_msg = format!("Not found archive id: {}", &file_object.id);

            // log
            error!(target: "stdout", "{}", &err_msg);

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
            error!(target: "stdout", "{}", &err_msg);

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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        info!(target: "stdout", "Open and read the file.");

        // open the file
        let mut file = match File::open(&file_path) {
            Ok(file) => file,
            Err(e) => {
                let err_msg = format!("Failed to open `{}`. {}", &file_object.filename, e);

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        // read the file
        let mut contents = String::new();
        if let Err(e) = file.read_to_string(&mut contents) {
            let err_msg = format!("Failed to read `{}`. {}", &file_object.filename, e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }

        info!(target: "stdout", "Chunk the file contents.");

        match chunk_text(&contents, extension, chunk_capacity) {
            Ok(chunks) => chunks,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    };

    // create index for chunks
    let mut index_response: Option<IndexResponse> = None;
    if !kw_search_url.is_empty() {
        let index_url = format!("{}/v1/index", &kw_search_url);
        info!(target: "stdout", "index_url: {}", &index_url);

        let mut index_request = IndexRequest { documents: vec![] };
        for chunk in chunks.iter() {
            let document_input = DocumentInput {
                content: chunk.clone(),
                title: None,
            };
            index_request.documents.push(document_input);
        }

        info!(target: "stdout", "Sending index request to kw-search-server");

        if let Ok(response) = reqwest::Client::new()
            .post(&index_url)
            .json(&index_request)
            .send()
            .await
        {
            if let Ok(idx_response) = response.json::<IndexResponse>().await {
                index_response = Some(idx_response);
            }
        }
    }

    // compute embeddings for chunks
    let embeddings_response = {
        // get the name of embedding model
        let model = match llama_core::utils::embedding_model_names() {
            Ok(model_names) => model_names[0].clone(),
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        };

        info!(target: "stdout", "Prepare the rag embedding request.");

        let api_key = match vdb_api_key.is_empty() {
            true => None,
            false => Some(vdb_api_key),
        };

        // create an embedding request
        let embedding_request = EmbeddingRequest {
            model: Some(model),
            input: chunks.into(),
            encoding_format: None,
            user: None,
            vdb_server_url: Some(vdb_server_url),
            vdb_collection_name: Some(vdb_collection_name),
            vdb_api_key: api_key,
        };

        match rag_doc_chunks_to_embeddings(&embedding_request).await {
            Ok(embedding_response) => embedding_response,
            Err(e) => {
                let err_msg = e.to_string();

                // log
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    };

    // create the create rag response
    let create_rag_response = if index_response.is_some() {
        CreateRagResponse {
            index_response,
            embeddings_response,
        }
    } else {
        CreateRagResponse {
            index_response: None,
            embeddings_response,
        }
    };

    // serialize embedding response
    let res = match serde_json::to_string(&create_rag_response) {
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
                    error!(target: "stdout", "{}", &err_msg);

                    error::internal_server_error(err_msg)
                }
            }
        }
        Err(e) => {
            let err_msg = format!("Fail to serialize embedding object. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the doc_to_embeddings response");

    res
}

pub(crate) async fn server_info_handler() -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming server info request.");

    // get the server info
    let server_info = match SERVER_INFO.get() {
        Some(server_info) => server_info.read().await,
        None => {
            let err_msg = "The server info is not set.";

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error("The server info is not set.");
        }
    };

    // serialize server info
    let s = match serde_json::to_string(&*server_info) {
        Ok(s) => s,
        Err(e) => {
            let err_msg = format!("Fail to serialize server info. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

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
            error!(target: "stdout", "{}", &err_msg);

            error::internal_server_error(err_msg)
        }
    };

    info!(target: "stdout", "Send the server info response.");

    res
}

pub(crate) async fn retrieve_handler(mut req: Request<Body>) -> Response<Body> {
    // log
    info!(target: "stdout", "Handling the coming retrieve request.");

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
                error!(target: "stdout", "{}", &err_msg);

                return error::internal_server_error(err_msg);
            }
        }
    }

    info!(target: "stdout", "Prepare the chat completion request.");

    // parse request
    let body_bytes = match to_bytes(req.body_mut()).await {
        Ok(body_bytes) => body_bytes,
        Err(e) => {
            let err_msg = format!("Fail to read buffer from request body. {}", e);

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::internal_server_error(err_msg);
        }
    };
    let mut chat_request: ChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
        Ok(chat_request) => chat_request,
        Err(e) => {
            let mut err_msg = format!("Fail to deserialize chat completion request: {}.", e);

            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                err_msg = format!("{}\njson_value: {}", err_msg, json_value);
            }

            // log
            error!(target: "stdout", "{}", &err_msg);

            return error::bad_request(err_msg);
        }
    };

    if chat_request.user.is_none() {
        chat_request.user = Some(gen_chat_id())
    };
    let id = chat_request.user.clone().unwrap();

    // log user id
    info!(target: "stdout", "user: {}", &id);

    // qdrant config
    let qdrant_config_vec = match get_qdrant_configs(&chat_request).await {
        Ok(qdrant_config_vec) => qdrant_config_vec,
        Err(e) => return error::internal_server_error(e.to_string()),
    };

    // retrieve context
    let retrieve_object_vec = match retrieve_context_with_multiple_qdrant_configs(
        &chat_request,
        &qdrant_config_vec,
    )
    .await
    {
        Ok(retrieve_object_vec) => retrieve_object_vec,
        Err(response) => {
            return response;
        }
    };

    // log retrieve object
    debug!(target: "stdout", "retrieve_object_vec:\n{}", serde_json::to_string_pretty(&retrieve_object_vec).unwrap());

    let res = {
        // serialize retrieve object
        let s = match serde_json::to_string(&retrieve_object_vec) {
            Ok(s) => s,
            Err(e) => {
                let err_msg = format!("Fail to serialize retrieve object. {}", e);

                // log
                error!(target: "stdout", "{}", &err_msg);

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
                error!(target: "stdout", "{}", &err_msg);

                error::internal_server_error(e.to_string())
            }
        }
    };

    info!(target: "stdout", "Send the retrieve response.");

    res
}

async fn get_qdrant_configs(
    chat_request: &ChatCompletionRequest,
) -> Result<Vec<QdrantConfig>, error::ServerError> {
    match (
        chat_request.vdb_server_url.as_deref(),
        chat_request.vdb_collection_name.as_deref(),
        chat_request.limit.as_deref(),
        chat_request.score_threshold.as_deref(),
    ) {
        (Some(url), Some(collection_name), Some(limit), Some(score_threshold)) => {
            // check if the length of collection name, limit, score_threshold are same
            if collection_name.len() != limit.len()
                || collection_name.len() != score_threshold.len()
            {
                let err_msg =
                    "The number of elements of `collection name`, `limit`, `score_threshold` in the request should be same.";

                // log
                error!(target: "stdout", "{}", &err_msg);

                return Err(error::ServerError::Operation(err_msg.into()));
            }

            info!(target: "stdout", "use the VectorDB settings from the request.");

            let collection_name_str = collection_name.join(",");
            let limit_str = limit
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<String>>()
                .join(",");
            let score_threshold_str = score_threshold
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<String>>()
                .join(",");
            info!(target: "stdout", "qdrant url: {}, collection name: {}, limit: {}, score threshold: {}", url, collection_name_str, limit_str, score_threshold_str);

            let mut qdrant_config_vec = vec![];
            for (idx, col_name) in collection_name.iter().enumerate() {
                qdrant_config_vec.push(QdrantConfig {
                    url: url.to_string(),
                    collection_name: col_name.to_string(),
                    limit: limit[idx],
                    score_threshold: score_threshold[idx],
                });
            }

            Ok(qdrant_config_vec)
        }
        (None, None, None, None) => {
            info!(target: "stdout", "use the default VectorDB settings.");

            let qdrant_config_vec = SERVER_INFO
                .get()
                .unwrap()
                .read()
                .await
                .qdrant_config
                .clone();

            Ok(qdrant_config_vec)
        }
        _ => {
            let err_msg = "The VectorDB settings in the request are not correct. The `url_vdb_server`, `collection_name`, `limit`, `score_threshold` fields in the request should be provided. The number of elements of `collection name`, `limit`, `score_threshold` should be same.";

            error!(target: "stdout", "{}", &err_msg);

            Err(error::ServerError::Operation(err_msg.into()))
        }
    }
}

fn calculate_hash(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn normalize(scores: &HashMap<u64, f32>) -> HashMap<u64, f32> {
    let min_score = scores.values().cloned().fold(f32::INFINITY, f32::min);
    let max_score = scores.values().cloned().fold(f32::NEG_INFINITY, f32::max);
    scores
        .iter()
        .map(|(&doc_id, &score)| {
            let normalized_score = if max_score - min_score > 0.0 {
                (score - min_score) / (max_score - min_score)
            } else {
                0.0
            };
            (doc_id, normalized_score)
        })
        .collect()
}

fn weighted_fusion(
    bm25_scores: HashMap<u64, f32>,
    embedding_scores: HashMap<u64, f32>,
    alpha: f32,
) -> HashMap<u64, f32> {
    // Normalize BM25 and Embedding scores
    let bm25_normalized = normalize(&bm25_scores);
    let embedding_normalized = normalize(&embedding_scores);

    // Get the union of all document IDs
    let all_doc_ids: HashSet<u64> = bm25_scores
        .keys()
        .chain(embedding_scores.keys())
        .cloned()
        .collect();

    // Calculate fusion scores
    all_doc_ids
        .into_iter()
        .map(|doc_id| {
            let bm25_score = *bm25_normalized.get(&doc_id).unwrap_or(&0.0);
            let embedding_score = *embedding_normalized.get(&doc_id).unwrap_or(&0.0);
            let final_score = alpha * bm25_score + (1.0 - alpha) * embedding_score;
            (doc_id, final_score)
        })
        .collect()
}
