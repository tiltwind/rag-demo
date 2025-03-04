from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility


urls = [
    "https://milvus.io/docs/glossary.md",
    "https://milvus.io/docs/architecture_overview.md",
]
loader = WebBaseLoader(urls)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

print(f"split docs: {len(split_docs)}")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

texts = [doc.page_content for doc in split_docs]
embeddings = model.encode(texts, convert_to_numpy=True)

print("embedding finish")

connections.connect(host='10.225.51.84', port='19530')

collection = Collection(name="rag_docs")

data = [
    embeddings.tolist(),
    texts
]
collection.insert(data)

print("insert data finish")


