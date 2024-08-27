use crate::error::ServerError;
use llama_core::search::{SearchOutput, SearchResult};
use serde::Serialize;

// Note: bing also requires the `Ocp-Apim-Subscription-Key` header: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/headers

#[allow(non_snake_case)]
#[derive(Serialize)]
pub struct BingSearchInput {
    /// The number of search results to return in the response. The default is 10 and the maximum value is 50. The actual number delivered may be less than requested.
    pub count: u8,
    /// The user's search query term. The term may not be empty.
    pub q: String,
    /// FIlter list for responses useful to the LLM.
    pub responseFilter: String,
}

#[allow(dead_code)]
pub fn bing_parser(
    raw_results: &serde_json::Value,
) -> Result<SearchOutput, Box<dyn std::error::Error>> {
    println!("\n\n\n RAW RESULTS: \n\n\n {}", raw_results.to_string());

    // parse webpages
    let web_pages_object = match raw_results["webPages"].is_object() {
        true => match raw_results["webPages"]["value"].as_array() {
            Some(value) => value,
            None => {
                let msg = r#"could not convert the "value" field of "webPages" to an array"#;
                error!(target: "bing_parser", "bing_parser: {}", msg);
                return Err(Box::new(ServerError::SearchConversionError(
                    msg.to_string(),
                )));
            }
        },
        false => {
            let msg = "no webpages found when parsing query.";
            error!(target: "bing_parser", "bing_parser: {}", msg);
            return Err(Box::new(ServerError::SearchConversionError(
                msg.to_string(),
            )));
        }
    };

    let mut results = Vec::new();
    for result in web_pages_object {
        let current_result = SearchResult {
            url: result["url"].to_string(),
            site_name: result["siteName"].to_string(),
            text_content: result["snippet"].to_string(),
        };
        results.push(current_result);
    }

    Ok(SearchOutput { results })
}
