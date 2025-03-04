from sentence_transformers import SentenceTransformer
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
import requests

connections.connect(host='10.225.51.84', port='19530')
collection = Collection(name="rag_docs")
collection.load()
print("collection load")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print(model)

query_text = "what's milvus?"
query_embedding = model.encode([query_text], convert_to_numpy=True).tolist()[0]

search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["text"]
)

contexts = []
for result in results:
    for hit in result:
        contexts.append(hit.entity.get('text'))


# contexts = [hit.entity.get('text') for hit in results[0]]

context_str = "\n\n".join([f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)])

prompt=f"""请基于以下上下文信息回答问题。如果上下文不包含答案或信息不足，请直接回答你不知道。

{context_str}

[问题]: {query_text}

请用中文给出清晰、简洁的回答，并确保回答完全基于提供的上下文。"""

print(prompt)

deep_api_key = "sk-8afe1e96beb84883bfee168a155cc21a"

response = requests.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {deep_api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的AI助手，能够准确根据提供的信息回答问题"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 8192
    },
    timeout=30
)

responseJson = response.json()

print(responseJson["choices"][0]["message"]["content"])
