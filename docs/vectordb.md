# Interaction with VectorDB

LlamaEdge-RAG interacts with external VectorDB through two approaches: one is via the CLI options of rag-api-server, and the other is through the request fields. In the following two sections, these two approaches are discussed separately. For the convenience of the following discussion, Qdrant is used as the example VectorDB.

## Via CLI options

If retrieving information from a fixed VectorDB, this method is recommended. The startup command of rag-api-server provides four command-line options, which are:

- `--qdrant-url <QDRANT_URL>` specifies the URL of VectorDB REST Service
- `--qdrant-collection-name <QDRANT_COLLECTION_NAME>` specifies one or multiple names of VectorDB collections
- `--qdrant-limit <QDRANT_LIMIT>` specifies the max number of retrieved result (no less than 1) from each collection specified in the `--qdrant-collection-name` option
- `--qdrant-score-threshold <QDRANT_SCORE_THRESHOLD>` specifies the minimal score threshold for the search results from each collection specified in the `--qdrant-collection-name` option

By setting the above four options in the startup command when starting rag-api-server, it helps avoid repeatedly providing these parameters in every retrieval request, such as chat completion request. The following is an example of the startup command:

```bash
wasmedge --dir .:. \
--env VDB_API_KEY=your-vdb-api-key \
...
--qdrant-url https://651ca7e5-e1d1-4851-abba-xxxxxxxxxxxx.europe-west3-0.gcp.cloud.qdrant.io:6333 \
--qdrant-collection-name paris1,paris2 \
--qdrant-limit 3,5 \
--qdrant-score-threshold 0.5,0.7
```

**Note** that `--env VDB_API_KEY=your-vdb-api-key` is required if the VectorDB requires an API key for access.

## Via request fields

For the cases where retrieving information from different VectorDBs or collections in different requests, this method is recommended. The requests for chat completions and rag creation tasks provide the fields respectively for specifying the VectorDB information.

### VectorDB related fields in requests to the `/v1/create/rag` endpoint

The request to the `/v1/create/rag` endpoint also provides the fields for specifying the VectorDB information, which are

- `vdb_server_url` specifies the URL of VectorDB REST Service
- `vdb_collection_name` specifies one or multiple names of VectorDB collections
- `vdb_api_key` specifies the API key for accessing the VectorDB

The following is an example of the request:

```bash
curl --location 'http://localhost:8080/v1/create/rag' \
--header 'Content-Type: multipart/form-data' \
--form 'file=@"paris.txt"' \
--form 'vdb_server_url="https://651ca7e5-e1d1-4851-abba-xxxxxxxxxxxx.europe-west3-0.gcp.cloud.qdrant.io:6333"' \
--form 'vdb_collection_name="paris"' \
--form 'vdb_api_key="your-vdb-api-key"'
```

### VectorDB related fields in the request to the `/v1/chat/completion` endpoint

The chat completion request to the `/v1/chat/completion` endpoint defines five VectorDB related fields for specifying the VectorDB information, which are

- `vdb_server_url` specifies the URL of VectorDB REST Service
- `vdb_collection_name` specifies one or multiple names of VectorDB collections
- `limit` specifies the max number of retrieved result (no less than 1) from each collection specified in the `vdb_collection_name` field
- `score_threshold` specifies the minimal score threshold for the search results from each collection specified in the `vdb_collection_name` field
- `vdb_api_key` specifies the API key for accessing the VectorDB

The following is an example of the chat completion request:

```bash
curl --location 'http://localhost:8080/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions as concisely as possible."
        },
        {
            "role": "user",
            "content": "What is the location of Paris, France along the Seine river?"
        }
    ],
    "vdb_server_url": "https://651ca7e5-e1d1-4851-abba-xxxxxxxxxxxx.europe-west3-0.gcp.cloud.qdrant.io:6333",
    "vdb_collection_name": ["paris1","paris2"],
    "limit": [3,5],
    "score_threshold": [0.5,0.7],
    "vdb_api_key": "your-vdb-api-key",
    "model": "Llama-3.2-3B-Instruct",
    "stream": false
}'
```

**Note** that the `limit`, and `score_threshold` fields are required in the chat completion request if `vdb_server_url` and `vdb_collection_name` are present. The `vdb_api_key` field is required only if the VectorDB requires an API key for access.

### VectorDB related fields in the request to the `/v1/retrieve` endpoint

Similarly, the request to the `/v1/retrieve` endpoint defines five VectorDB related fields for specifying the VectorDB information, which are

- `vdb_server_url` specifies the URL of VectorDB REST Service
- `vdb_collection_name` specifies one or multiple names of VectorDB collections
- `limit` specifies the max number of retrieved result (no less than 1) from each collection specified in the `vdb_collection_name` field
- `score_threshold` specifies the minimal score threshold for the search results from each collection specified in the `vdb_collection_name` field
- `vdb_api_key` specifies the API key for accessing the VectorDB

The following is an example of the retrieval request:

```bash
curl --location 'http://localhost:8080/v1/retrieve' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions as concisely as possible."
        },
        {
            "role": "user",
            "content": "What is the location of Paris, France along the Seine river?"
        }
    ],
    "vdb_server_url": "https://651ca7e5-e1d1-4851-abba-xxxxxxxxxxxx.europe-west3-0.gcp.cloud.qdrant.io:6333",
    "vdb_collection_name": ["paris1","paris2"],
    "limit": [3,5],
    "score_threshold": [0.5,0.7],
    "vdb_api_key": "your-vdb-api-key",
    "model": "Llama-3.2-3B-Instruct",
    "stream": false
}'
```

**Note** that the `limit`, and `score_threshold` fields are required in the chat completion request if `vdb_server_url` and `vdb_collection_name` are present. The `vdb_api_key` field is required only if the VectorDB requires an API key for access.
