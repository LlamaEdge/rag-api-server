use crate::error::ServerError;
use llama_core::search::{SearchOutput, SearchResult};
use serde::Serialize;

#[allow(non_snake_case)]
#[derive(Serialize)]
pub struct TavilySearchInput {
    pub api_key: String,
    pub include_answer: bool,
    pub include_images: bool,
    pub query: String,
    pub max_results: u8,
    pub include_raw_content: bool,
    pub search_depth: String,
}

#[allow(dead_code)]
pub fn tavily_parser(
    raw_results: &serde_json::Value,
) -> Result<SearchOutput, Box<dyn std::error::Error>> {
    let results_array = match raw_results["results"].as_array() {
        Some(array) => array,
        None => {
            let msg = "No results returned from server";
            error!(target: "search_server", "google_parser: {}", msg);
            return Err(Box::new(ServerError::SearchConversionError(
                msg.to_string(),
            )));
        }
    };

    let mut results = Vec::new();

    for result in results_array {
        let current_result = SearchResult {
            url: result["url"].to_string(),
            site_name: result["title"].to_string(),
            text_content: result["content"].to_string(),
        };
        results.push(current_result)
    }

    Ok(SearchOutput { results })
}
