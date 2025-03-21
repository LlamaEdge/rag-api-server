# LlamaEdge-RAG API Server

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [LlamaEdge-RAG API Server](#llamaedge-rag-api-server)
  - [Introduction](#introduction)
    - [Endpoints](#endpoints)
      - [List models](#list-models)
      - [Chat completions](#chat-completions)
      - [Upload a file](#upload-a-file)
      - [List all files](#list-all-files)
      - [Retrieve information about a specific file](#retrieve-information-about-a-specific-file)
      - [Retrieve the content of a specific file](#retrieve-the-content-of-a-specific-file)
      - [Download a specific file](#download-a-specific-file)
      - [Delete a file](#delete-a-file)
      - [Segment a file to chunks](#segment-a-file-to-chunks)
      - [Compute embeddings for user query or file chunks](#compute-embeddings-for-user-query-or-file-chunks)
      - [Generate embeddings from a file](#generate-embeddings-from-a-file)
      - [Get server information](#get-server-information)
      - [Retrieve context](#retrieve-context)
  - [Setup](#setup)
  - [Build](#build)
  - [Execute](#execute)
  - [Usage Example](#usage-example)
  - [Set Log Level](#set-log-level)

<!-- /code_chunk_output -->

## Introduction

LlamaEdge-RAG API server provides a group of OpenAI-compatible web APIs for the Retrieval-Augmented Generation (RAG) applications. The server is implemented in WebAssembly (Wasm) and runs on [WasmEdge Runtime](https://github.com/WasmEdge/WasmEdge).

### Endpoints

#### List models

`rag-api-server` provides a POST API `/v1/models` to list currently available models.

<details> <summary> Example </summary>

You can use `curl` to test it on a new terminal:

```bash
curl -X POST http://localhost:8080/v1/models -H 'accept:application/json'
```

If the command runs successfully, you should see the similar output as below in your terminal:

```json
{
    "object":"list",
    "data":[
        {
            "id":"llama-2-chat",
            "created":1697084821,
            "object":"model",
            "owned_by":"Not specified"
        }
    ]
}
```

</details>

#### Chat completions

Ask a question using OpenAI's JSON message format.

<details> <summary> Example </summary>

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "Who is Robert Oppenheimer?"}], "model":"llama-2-chat"}'
```

Here is the response.

```json
{
    "id":"",
    "object":"chat.completion",
    "created":1697092593,
    "model":"llama-2-chat",
    "choices":[
        {
            "index":0,
            "message":{
                "role":"assistant",
                "content":"Robert Oppenheimer was an American theoretical physicist and director of the Manhattan Project, which developed the atomic bomb during World War II. He is widely regarded as one of the most important physicists of the 20th century and is known for his contributions to the development of quantum mechanics and the theory of the atomic nucleus. Oppenheimer was also a prominent figure in the post-war nuclear weapons debate, advocating for international control and regulation of nuclear weapons."
            },
            "finish_reason":"stop"
        }
    ],
    "usage":{
        "prompt_tokens":9,
        "completion_tokens":12,
        "total_tokens":21
    }
}
```

</details>

#### Upload a file

In RAG applications, uploading files is a necessary step.

<details> <summary> Example: Upload a file </summary>

The following command upload a text file [paris.txt](https://huggingface.co/datasets/gaianet/paris/raw/main/paris.txt) to the API server via the `/v1/files` endpoint:

```bash
curl -X POST http://127.0.0.1:8080/v1/files -F "file=@paris.txt"
```

If the command is successful, you should see the similar output as below in your terminal:

```json
{
    "id": "file_4bc24593-2a57-4646-af16-028855e7802e",
    "bytes": 2161,
    "created_at": 1711611801,
    "filename": "paris.txt",
    "object": "file",
    "purpose": "assistants"
}
```

The `id` and `filename` fields are important for the next step, for example, to segment the uploaded file to chunks for computing embeddings.

</details>

#### List all files

`GET /v1/files` endpoint is used for listing all files on the server.

<details> <summary> Example: List files </summary>

The following command lists all files on the server via the `/v1/files` endpoint:

```bash
curl -X GET http://127.0.0.1:8080/v1/files
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "object": "list",
    "data": [
        {
            "id": "file_33d9188d-5060-4141-8c52-ae148fd15f6a",
            "bytes": 17039,
            "created_at": 1718296362,
            "filename": "test-123.m4a",
            "object": "file",
            "purpose": "assistants"
        },
        {
            "id": "file_8c6439da-df59-4b9a-bb5e-dba4b2f23c04",
            "bytes": 17039,
            "created_at": 1718294169,
            "filename": "test-123.m4a",
            "object": "file",
            "purpose": "assistants"
        }
    ]
}
```

</details>

#### Retrieve information about a specific file

`GET /v1/files/{file_id}` endpoint is used for retrieving information about a specific file on the server.

<details> <summary> Example: Retrieve information about a specific file </summary>

The following command retrieves information about a specific file on the server via the `/v1/files/{file_id}` endpoint:

```bash
curl -X GET http://localhost:10086/v1/files/file_b892bc81-35e9-44a6-8c01-ae915c1d3832
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "id": "file_b892bc81-35e9-44a6-8c01-ae915c1d3832",
    "bytes": 2161,
    "created_at": 1715832065,
    "filename": "paris.txt",
    "object": "file",
    "purpose": "assistants"
}
```

</details>

#### Retrieve the content of a specific file

`GET /v1/files/{file_id}/content` endpoint is used for retrieving the content of a specific file on the server.

<details> <summary> Example: Retrieve the content of a specific file </summary>

The following command retrieves the content of a specific file on the server via the `/v1/files/{file_id}/content` endpoint:

```bash
curl -X GET http://localhost:10086/v1/files/file_b892bc81-35e9-44a6-8c01-ae915c1d3832/content
```

</details>

#### Download a specific file

`GET /v1/files/download/{file_id}` endpoint is used for downloading a specific file on the server.

<details> <summary> Example: Download a specific file </summary>

The following command downloads a specific file on the server via the `/v1/files/download/{file_id}` endpoint:

```bash
curl -X GET http://localhost:10086/v1/files/download/file_b892bc81-35e9-44a6-8c01-ae915c1d3832
```

</details>

#### Delete a file

`DELETE /v1/files/{file_id}` endpoint is used for deleting a specific file on the server.

<details> <summary> Example: Delete a specific file </summary>

The following command deletes a specific file on the server via the `/v1/files/{file_id}` endpoint:

```bash
curl -X DELETE http://localhost:10086/v1/files/file_6a6d8046-fd98-410a-b70e-0a0142ec9a39
```

If the command is successful, you should see the similar output as below in your terminal:

```bash
{
    "id": "file_6a6d8046-fd98-410a-b70e-0a0142ec9a39",
    "object": "file",
    "deleted": true
}
```

</details>

#### Segment a file to chunks

To segment the uploaded file to chunks for computing embeddings, use the `/v1/chunks` API.

<details> <summary> Example </summary>

The following command sends the uploaded file ID and filename to the API server and gets the chunks:

```bash
curl -X POST http://localhost:8080/v1/chunks \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id":"file_4bc24593-2a57-4646-af16-028855e7802e", "filename":"paris.txt"}'
```

The following is an example return with the generated chunks:

```json
{
    "id": "file_4bc24593-2a57-4646-af16-028855e7802e",
    "filename": "paris.txt",
    "chunks": [
        "Paris, city and capital of France, ..., for Paris has retained its importance as a centre for education and intellectual pursuits.",
        "Paris’s site at a crossroads ..., drawing to itself much of the talent and vitality of the provinces."
    ]
}
```

</details>

#### Compute embeddings for user query or file chunks

To compute embeddings for user query or file chunks, use the `/v1/embeddings` API.

<details> <summary> Example </summary>

The following command sends a query to the API server and gets the embeddings as return:

```bash
curl -X POST http://localhost:8080/v1/embeddings \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"model": "e5-mistral-7b-instruct-Q5_K_M", "input":["Paris, city and capital of France, ..., for Paris has retained its importance as a centre for education and intellectual pursuits.", "Paris’s site at a crossroads ..., drawing to itself much of the talent and vitality of the provinces."]}'
```

The embeddings returned are like below:

```json
{
    "object": "list",
    "data": [
        {
            "index": 0,
            "object": "embedding",
            "embedding": [
                0.1428378969,
                -0.0447309874,
                0.007660218049,
                ...
                -0.0128974719,
                -0.03543198109,
                0.03974733502,
                0.00946635101,
                -0.01531364303
            ]
        },
        {
            "index": 1,
            "object": "embedding",
            "embedding": [
                0.0697753951,
                -0.0001159032545,
                0.02073983476,
                ...
                0.03565846011,
                -0.04550019652,
                0.02691745944,
                0.02498772368,
                -0.003226313973
            ]
        }
    ],
    "model": "e5-mistral-7b-instruct-Q5_K_M",
    "usage": {
        "prompt_tokens": 491,
        "completion_tokens": 0,
        "total_tokens": 491
    }
}
```

</details>

#### Generate embeddings from a file

`/v1/create/rag` endpoint provides users a one-click way to convert a text or markdown file to embeddings directly. The effect of the endpoint is equivalent to running `/v1/files` + `/v1/chunks` + `/v1/embeddings` sequently. Note that the `--chunk-capacity` CLI option is required for the endpoint. The default value of the option is `100`. You can set it to different values while starting LlamaEdge-RAG API server.

<details> <summary> Example </summary>

The following command uploads a text file [paris.txt](https://huggingface.co/datasets/gaianet/paris/raw/main/paris.txt) to the API server via the `/v1/create/rag` endpoint:

```bash
curl -X POST http://127.0.0.1:8080/v1/create/rag -F "file=@paris.txt"
```

The embeddings returned are like below:

```json
{
    "object": "list",
    "data": [
        {
            "index": 0,
            "object": "embedding",
            "embedding": [
                0.1428378969,
                -0.0447309874,
                0.007660218049,
                ...
                -0.0128974719,
                -0.03543198109,
                0.03974733502,
                0.00946635101,
                -0.01531364303
            ]
        },
        {
            "index": 1,
            "object": "embedding",
            "embedding": [
                0.0697753951,
                -0.0001159032545,
                0.02073983476,
                ...
                0.03565846011,
                -0.04550019652,
                0.02691745944,
                0.02498772368,
                -0.003226313973
            ]
        }
    ],
    "model": "e5-mistral-7b-instruct-Q5_K_M",
    "usage": {
        "prompt_tokens": 491,
        "completion_tokens": 0,
        "total_tokens": 491
    }
}
```

</details>

#### Get server information

`/v1/info` endpoint provides the information of the API server, including the version of the server, the parameters of models, and etc.

<details> <summary> Example </summary>

You can use `curl` to test it on a new terminal:

```bash
curl -X POST http://localhost:8080/v1/info -H 'accept:application/json'
```

If the command runs successfully, you should see the similar output as below in your terminal:

```json
{
    "version": "0.3.4",
    "plugin_version": "b2694 (commit 0d56246f)",
    "port": "8080",
    "models": [
        {
            "name": "Llama-2-7b-chat-hf-Q5_K_M",
            "type": "chat",
            "prompt_template": "Llama2Chat",
            "n_predict": 1024,
            "n_gpu_layers": 100,
            "ctx_size": 4096,
            "batch_size": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        },
        {
            "name": "all-MiniLM-L6-v2-ggml-model-f16",
            "type": "embedding",
            "prompt_template": "Llama2Chat",
            "n_predict": 1024,
            "n_gpu_layers": 100,
            "ctx_size": 384,
            "batch_size": 512,
            "temperature": 1.0,
            "top_p": 1.0,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
    ],
    "qdrant_config": {
        "url": "http://localhost:6333",
        "collection_name": "default",
        "limit": 5,
        "score_threshold": 0.4
    }
}
```

</details>

#### Retrieve context

`/v1/retrieve` endpoint sends a query and gets the retrieval results.

<details> <summary> Example </summary>

You can use `curl` to test it on a new terminal:

```bash
curl -X POST http://localhost:8080/v1/retrieve \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the location of Paris, France along the Seine River?"}], "model":"llama-2-chat"}'
```

If the command runs successfully, you should see the similar output as below in your terminal:

```json
{
    "points": [
        {
            "source": "\"Paris is located in northern central France, in a north-bending arc of the river Seine whose crest includes two islands, the Île Saint-Louis and the larger Île de la Cité, which form the oldest part of the city. The river's mouth on the English Channel is about 233 mi downstream from the city. The city is spread widely on both banks of the river. Overall, the city is relatively flat, and the lowest point is 35 m above sea level. Paris has several prominent hills, the highest of which is Montmartre at 130 m.\\n\"",
            "score": 0.74011195
        },
        {
            "source": "\"The Paris region is the most active water transport area in France, with most of the cargo handled by Ports of Paris in facilities located around Paris. The rivers Loire, Rhine, Rhône, Me\\n\"",
            "score": 0.63990676
        },
        {
            "source": "\"Paris\\nCountry\\tFrance\\nRegion\\nÎle-de-France\\r\\nDepartment\\nParis\\nIntercommunality\\nMétropole du Grand Paris\\nSubdivisions\\n20 arrondissements\\nGovernment\\n • Mayor (2020–2026)\\tAnne Hidalgo (PS)\\r\\nArea\\n1\\t105.4 km2 (40.7 sq mi)\\n • Urban\\n (2020)\\t2,853.5 km2 (1,101.7 sq mi)\\n • Metro\\n (2020)\\t18,940.7 km2 (7,313.0 sq mi)\\nPopulation\\n (2023)\\n2,102,650\\n • Rank\\t9th in Europe\\n1st in France\\r\\n • Density\\t20,000/km2 (52,000/sq mi)\\n • Urban\\n (2019)\\n10,858,852\\n • Urban density\\t3,800/km2 (9,900/sq mi)\\n • Metro\\n (Jan. 2017)\\n13,024,518\\n • Metro density\\t690/km2 (1,800/sq mi)\\nDemonym(s)\\nParisian(s) (en) Parisien(s) (masc.), Parisienne(s) (fem.) (fr), Parigot(s) (masc.), \\\"Parigote(s)\\\" (fem.) (fr, colloquial)\\nTime zone\\nUTC+01:00 (CET)\\r\\n • Summer (DST)\\nUTC+02:00 (CEST)\\r\\nINSEE/Postal code\\t75056 /75001-75020, 75116\\r\\nElevation\\t28–131 m (92–430 ft)\\n(avg. 78 m or 256 ft)\\nWebsite\\twww.paris.fr\\r\\n1 French Land Register data, which excludes lakes, ponds, glaciers > 1 km2 (0.386 sq mi or 247 acres) and river estuaries.\\n\"",
            "score": 0.62259054
        },
        {
            "source": "\" in Paris\\n\"",
            "score": 0.6152092
        },
        {
            "source": "\"The Parisii, a sub-tribe of the Celtic Senones, inhabited the Paris area from around the middle of the 3rd century BC. One of the area's major north–south trade routes crossed the Seine on the île de la Cité, which gradually became an important trading centre. The Parisii traded with many river towns (some as far away as the Iberian Peninsula) and minted their own coins.\\n\"",
            "score": 0.5720232
        }
    ],
    "limit": 5,
    "score_threshold": 0.4
}
```

</details>

## Setup

Llama-RAG API server runs on WasmEdge Runtime. According to the operating system you are using, choose the installation command:

<details> <summary> For macOS (apple silicon) </summary>

```console
# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s

# Assuming you use zsh (the default shell on macOS), run the following command to activate the environment
source $HOME/.zshenv
```

</details>

<details> <summary> For Ubuntu (>= 20.04) </summary>

```console
# install libopenblas-dev
apt update && apt install -y libopenblas-dev

# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

<details> <summary> For General Linux </summary>

```console
# install WasmEdge-0.13.4 with wasi-nn-ggml plugin
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s

# Assuming you use bash (the default shell on Ubuntu), run the following command to activate the environment
source $HOME/.bashrc
```

</details>

## Build

```bash
# Clone the repository
git clone https://github.com/LlamaEdge/rag-api-server.git

# Change the working directory
cd rag-api-server

# (Optional) Add the `wasm32-wasip1` target to the Rust toolchain
rustup target add wasm32-wasip1

# Build `rag-api-server.wasm` with the `http` support only, or
cargo build --target wasm32-wasip1 --release

# Build `rag-api-server.wasm` with both `http` and `https` support
cargo build --target wasm32-wasip1 --release --features full

# Copy the `rag-api-server.wasm` to the root directory
cp target/wasm32-wasip1/release/rag-api-server.wasm .
```

To check the CLI options of the `rag-api-server` wasm app, you can run the following command:

  ```bash
  $ wasmedge rag-api-server.wasm -h

  LlamaEdge-RAG API Server

  Usage: rag-api-server.wasm [OPTIONS] --model-name <MODEL_NAME> --prompt-template <PROMPT_TEMPLATE>

Options:
  -m, --model-name <MODEL_NAME>
          Sets names for chat and embedding models. The names are separated by comma without space, for example, '--model-name Llama-2-7b,all-minilm'

  -a, --model-alias <MODEL_ALIAS>
          Model aliases for chat and embedding models

          [default: default,embedding]

  -c, --ctx-size <CTX_SIZE>
          Sets context sizes for chat and embedding models, respectively. The sizes are separated by comma without space, for example, '--ctx-size 4096,384'. The first value is for the chat model, and the second is for the embedding model

          [default: 4096,384]

  -p, --prompt-template <PROMPT_TEMPLATE>
          Sets prompt templates for chat and embedding models, respectively. The prompt templates are separated by comma without space, for example, '--prompt-template llama-2-chat,embedding'. The first value is for the chat model, and the second is for the embedding model

          [possible values: llama-2-chat, llama-3-chat, llama-3-tool, mistral-instruct, mistral-tool, mistrallite, mistral-small-chat, mistral-small-tool, openchat, codellama-instruct, codellama-super-instruct, human-assistant, vicuna-1.0-chat, vicuna-1.1-chat, vicuna-llava, chatml, chatml-tool, internlm-2-tool, baichuan-2, wizard-coder, zephyr, stablelm-zephyr, intel-neural, deepseek-chat, deepseek-coder, deepseek-chat-2, deepseek-chat-25, deepseek-chat-3, solar-instruct, phi-2-chat, phi-2-instruct, phi-3-chat, phi-3-instruct, phi-4-chat, gemma-instruct, gemma-3, octopus, glm-4-chat, groq-llama3-tool, mediatek-breeze, nemotron-chat, nemotron-tool, functionary-32, functionary-31, minicpmv, moxin-chat, falcon3, megrez, qwen2-vision, exaone-deep-chat, exaone-chat, embedding, tts, none]

  -r, --reverse-prompt <REVERSE_PROMPT>
          Halt generation at PROMPT, return control

  -n, --n-predict <N_PREDICT>
          Number of tokens to predict, -1 = infinity, -2 = until context filled

          [default: -1]

  -g, --n-gpu-layers <N_GPU_LAYERS>
          Number of layers to run on the GPU

          [default: 100]

      --split-mode <SPLIT_MODE>
          Split the model across multiple GPUs. Possible values: `none` (use one GPU only), `layer` (split layers and KV across GPUs, default), `row` (split rows across GPUs)

          [default: layer]

      --main-gpu <MAIN_GPU>
          The main GPU to use

      --tensor-split <TENSOR_SPLIT>
          How split tensors should be distributed accross GPUs. If None the model is not split; otherwise, a comma-separated list of non-negative values, e.g., "3,2" presents 60% of the data to GPU 0 and 40% to GPU 1

      --threads <THREADS>
          Number of threads to use during computation

          [default: 2]

      --grammar <GRAMMAR>
          BNF-like grammar to constrain generations (see samples in grammars/ dir)

          [default: ]

      --json-schema <JSON_SCHEMA>
          JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object. For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead

  -b, --batch-size <BATCH_SIZE>
          Sets batch sizes for chat and embedding models, respectively. The sizes are separated by comma without space, for example, '--batch-size 128,64'. The first value is for the chat model, and the second is for the embedding model

          [default: 512,512]

  -u, --ubatch-size <UBATCH_SIZE>
          Sets physical maximum batch sizes for chat and/or embedding models. To run both chat and embedding models, the sizes should be separated by comma without space, for example, '--ubatch-size 512,512'. The first value is for the chat model, and the second for the embedding model

          [default: 512,512]

      --rag-prompt <RAG_PROMPT>
          Custom rag prompt

      --rag-policy <POLICY>
          Strategy for merging RAG context into chat messages

          [default: system-message]

          Possible values:
          - system-message:    Merge RAG context into the system message
          - last-user-message: Merge RAG context into the last user message

      --qdrant-url <QDRANT_URL>
          URL of Qdrant REST Service

          [default: http://127.0.0.1:6333]

      --qdrant-collection-name <QDRANT_COLLECTION_NAME>
          Name of Qdrant collection

          [default: default]

      --qdrant-limit <QDRANT_LIMIT>
          Max number of retrieved result (no less than 1)

          [default: 5]

      --qdrant-score-threshold <QDRANT_SCORE_THRESHOLD>
          Minimal score threshold for the search result

          [default: 0.4]

      --chunk-capacity <CHUNK_CAPACITY>
          Maximum number of tokens each chunk contains

          [default: 100]

      --context-window <CONTEXT_WINDOW>
          Maximum number of user messages used in the retrieval

          [default: 1]

      --kw-search-url <KW_SEARCH_URL>
          URL of the keyword search service

      --include-usage
          Whether to include usage in the stream response. Defaults to false

      --socket-addr <SOCKET_ADDR>
          Socket address of LlamaEdge-RAG API Server instance. For example, `0.0.0.0:8080`

      --port <PORT>
          Port number

          [default: 8080]

      --web-ui <WEB_UI>
          Root path for the Web UI files

          [default: chatbot-ui]

      --log-prompts
          Deprecated. Print prompt strings to stdout

      --log-stat
          Deprecated. Print statistics to stdout

      --log-all
          Deprecated. Print all log information to stdout

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
  ```

</details>

## Execute

LlamaEdge-RAG API server requires two types of models: chat and embedding. The chat model is used for generating responses to user queries, while the embedding model is used for computing embeddings for user queries or file chunks.

Execution also requires the presence of a running [Qdrant](https://qdrant.tech/) service.

For the purpose of demonstration, we use the [Llama-2-7b-chat-hf-Q5_K_M.gguf](https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/Llama-2-7b-chat-hf-Q5_K_M.gguf) and [all-MiniLM-L6-v2-ggml-model-f16.gguf](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-ggml-model-f16.gguf) models as examples. Download these models and place them in the root directory of the repository.

- Ensure the Qdrant service is running

    ```bash
    # Pull the Qdrant docker image
    docker pull qdrant/qdrant

    # Create a directory to store Qdrant data
    mkdir qdrant_storage

    # Run Qdrant service
    docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
    ```

- Start an instance of LlamaEdge-RAG API server

  ```bash
  wasmedge --dir .:. --nn-preload default:GGML:AUTO:Llama-2-7b-chat-hf-Q5_K_M.gguf \
      --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
      rag-api-server.wasm \
      --model-name Llama-2-7b-chat-hf-Q5_K_M,all-MiniLM-L6-v2-ggml-model-f16 \
      --ctx-size 4096,384 \
      --prompt-template llama-2-chat,embedding \
      --rag-policy system-message \
      --qdrant-collection-name default \
      --qdrant-limit 3 \
      --qdrant-score-threshold 0.5 \
      --rag-prompt "Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n" \
      --port 8080
  ```

## Usage Example

- [Execute](#execute) the server

- Generate embeddings for [paris.txt](https://huggingface.co/datasets/gaianet/paris/raw/main/paris.txt) via the `/v1/create/rag` endpoint

    ```bash
    curl -X POST http://127.0.0.1:8080/v1/create/rag -F "file=@paris.txt"
    ```

- Ask a question

    ```bash
    curl -X POST http://localhost:8080/v1/chat/completions \
        -H 'accept:application/json' \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"system", "content": "You are a helpful assistant."}, {"role":"user", "content": "What is the location of Paris, France along the Seine River?"}], "model":"Llama-2-7b-chat-hf-Q5_K_M"}'
    ```

## Set Log Level

You can set the log level of the API server by setting the `LLAMA_LOG` environment variable. For example, to set the log level to `debug`, you can run the following command:

```bash
wasmedge --dir .:. --env LLAMA_LOG=debug \
    --nn-preload default:GGML:AUTO:Llama-2-7b-chat-hf-Q5_K_M.gguf \
    --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
    rag-api-server.wasm \
    --model-name Llama-2-7b-chat-hf-Q5_K_M,all-MiniLM-L6-v2-ggml-model-f16 \
    --ctx-size 4096,384 \
    --prompt-template llama-2-chat,embedding \
    --rag-prompt "Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n"
```

The log level can be one of the following values: `trace`, `debug`, `info`, `warn`, `error`. The default log level is `info`.
