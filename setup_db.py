from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# 1. Initialize a local Qdrant instance
# This will create a folder named 'qdrant_local_data' in your directory
client = QdrantClient(path="./qdrant_local_data")

# 2. Define the collection parameters
collection_name = "enterprise_docs"
vector_size = 384 # Standard size for the 'all-MiniLM-L6-v2' sentence-transformer model

# 3. Create the collection if it doesn't exist
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# 1. Initialize a local Qdrant instance
client = QdrantClient(path="./qdrant_local_data")

# 2. Define collection parameters
collection_name = "enterprise_docs"
vector_size = 384

try:
    # 3. Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")

finally:
    # 4. Explicitly close client
    client.close()  