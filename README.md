# LlamaEdge-RAG API Server

## Endpoints

- `/v1/rag/embeddings` endpoint for converting text to embeddings in the one-click way.

- `/v1/rag/query` endpoint for querying the RAG model. It is equivalent to the [`/v1/chat/completions` endpoint](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server#v1chatcompletions-endpoint-for-chat-completions) defined in LlamaEdge API server.

## CLI options for the API server

The `-h` or `--help` option can list the available options of the `rag-api-server` wasm app:

  ```console
  $ wasmedge rag-api-server.wasm -h

  Usage: rag-api-server.wasm [OPTIONS] --model-name <MODEL-NAME> --prompt-template <TEMPLATE>

  Options:
  -m, --model-name <MODEL-NAME>
          Sets names for chat and embedding models. The names are separated by comma without space, for example, '--model-name Llama-2-7b,all-minilm'.
  -a, --model-alias <MODEL-ALIAS>
          Sets model aliases [default: default,embedding]
  -c, --ctx-size <CTX_SIZE>
          Sets context sizes for chat and embedding models. The sizes are separated by comma without space, for example, '--ctx-size 4096,384'. The first value is for the chat model, and the second value is for the embedding model. [default: 4096,384]
  -r, --reverse-prompt <REVERSE_PROMPT>
          Halt generation at PROMPT, return control.
  -p, --prompt-template <TEMPLATE>
          Sets the prompt template. [possible values: llama-2-chat, codellama-instruct, codellama-super-instruct, mistral-instruct, mistrallite, openchat, human-assistant, vicuna-1.0-chat, vicuna-1.1-chat, vicuna-llava, chatml, baichuan-2, wizard-coder, zephyr, stablelm-zephyr, intel-neural, deepseek-chat, deepseek-coder, solar-instruct, gemma-instruct]
      --system-prompt <system_prompt>
          Sets global system prompt. [default: ]
      --qdrant-url <qdrant_url>
          Sets the url of Qdrant REST Service. [default: http://localhost:6333]
      --qdrant-collection-name <qdrant_collection_name>
          Sets the collection name of Qdrant. [default: default]
      --qdrant-limit <qdrant_limit>
          Max number of retrieved result. [default: 3]
      --qdrant-score-threshold <qdrant_score_threshold>
          Minimal score threshold for the search result [default: 0.4]
      --log-prompts
          Print prompt strings to stdout
      --log-stat
          Print statistics to stdout
      --log-all
          Print all log information to stdout
      --web-ui <WEB_UI>
          Root path for the Web UI files [default: chatbot-ui]
  -s, --socket-addr <IP:PORT>
          Sets the socket address [default: 0.0.0.0:8080]
  -h, --help
          Print help
  -V, --version
          Print version
  ```

  Please guarantee that the port is not occupied by other processes. If the port specified is available on your machine and the command is successful, you should see the following output in the terminal:

  ```console
  Listening on http://0.0.0.0:8080
  ```

  If the Web UI is ready, you can navigate to `http://127.0.0.1:8080` to open the chatbot, it will interact with the API of your server.
