pub mod bing_search;
pub mod local_google_search;
pub mod tavily_search;

use crate::{error, SEARCH_ARGUMENTS, SEARCH_CONFIG};
use endpoints::chat::{
    ChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionUserMessageContent, ContentPart,
};

#[allow(dead_code)]
pub(crate) async fn insert_search_results(
    chat_request: &mut ChatCompletionRequest,
) -> Result<(), hyper::Response<hyper::Body>> {
    let search_arguments = match SEARCH_ARGUMENTS.get() {
        Some(sa) => sa,
        None => {
            return Err(error::internal_server_error(
                "Failed to get `SEARCH_ARGUMENTS`. Was it set?",
            ));
        }
    };

    if let Some(ChatCompletionRequestMessage::User(ref message)) = chat_request.messages.last() {
        let search_config = match SEARCH_CONFIG.get() {
            Some(sc) => sc,
            None => {
                let err_msg = format!("Failed to obtain SEARCH_CONFIG. Was it set?");
                error!(target: "insert_search_results", "{}", &err_msg);

                return Err(error::internal_server_error(err_msg));
            }
        };
        info!(target: "insert_search_results", "performing search");

        let user_message_content = match message.content() {
            ChatCompletionUserMessageContent::Text(message) => message.to_owned(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut message: String = "".to_owned();
                for part in parts {
                    match part {
                        ContentPart::Text(message_part) => {
                            message.push_str(message_part.text());
                        }
                        ContentPart::Image(_) => {}
                    }
                }
                message
            }
        };

        // set search input.
        let search_input = tavily_search::TavilySearchInput {
            api_key: search_arguments.api_key.to_owned(),
            include_answer: false,
            include_images: false,
            query: user_message_content,
            max_results: search_config.max_search_results,
            include_raw_content: false,
            search_depth: "advanced".to_owned(),
        };

        // Prepare the final `results` string for use as input.
        let mut results = search_arguments.search_prompt.clone();

        match search_arguments.summarize {
            true => {
                match search_config.summarize_search(&search_input).await {
                    // Append the result summary to the search prompt.
                    Ok(search_summary) => results += search_summary.as_str(),
                    Err(e) => {
                        let err_msg = format!(
                            "Failed to performing summarized search on SEACH_CONFIG {msg}",
                            msg = e
                        );
                        error!(target: "insert_search_results", "{}", &err_msg);

                        return Err(error::internal_server_error(err_msg));
                    }
                };
            }
            false => {
                let search_output: llama_core::search::SearchOutput =
                    match search_config.perform_search(&search_input).await {
                        Ok(search_output) => search_output,
                        Err(e) => {
                            let err_msg =
                                format!("Failed to perform search on SEACH_CONFIG: {msg}", msg = e);
                            error!(target: "insert_search_results", "{}", &err_msg);

                            return Err(error::internal_server_error(err_msg));
                        }
                    };

                for result in search_output.results {
                    results.push_str(result.text_content.as_str());
                    results.push_str("\n\n");
                }
            }
        }

        let system_search_result_message = ChatCompletionSystemMessage::new(results, None);

        chat_request.messages.insert(
            chat_request.messages.len() - 1,
            ChatCompletionRequestMessage::System(system_search_result_message),
        )
    }
    Ok(())
}
