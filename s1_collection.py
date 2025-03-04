from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

connections.connect(host='10.225.51.84', port='19530')

# Dimension of the vector
dimensions = 768

# 定义字段 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]

# 创建 collection schema
schema = CollectionSchema(fields=fields, description="documents collection")

print(schema)

# 创建 collection
collection_name = "rag_docs"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)


print(collection)

# 创建索引
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index(field_name="embedding", index_params=index_params)