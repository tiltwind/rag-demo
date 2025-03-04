from sentence_transformers import SentenceTransformer
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

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

for result in results:
    for hit in result:
        print(f"Distance: {hit.distance}")
        print(f"Text: {hit.entity.get('text')}")
        print("-" * 50)

connections.disconnect("default")


