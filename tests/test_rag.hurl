
# test /v1/chat/completions endpoint
POST http://localhost:8080/v1/chat/completions
Accept: application/json
Content-Type: application/json
```json
{
    "messages": [
        {
            "role": "user",
            "content": "What is the location of Paris, France along the Siene River?"
        }
    ],
    "model": "Qwen2-1.5B-Instruct",
    "stream": false
}
```
HTTP 200
[Asserts]
jsonpath "$.model" == "Qwen2-1.5B-Instruct"
jsonpath "$.choices[0].message.content" contains "Paris"
