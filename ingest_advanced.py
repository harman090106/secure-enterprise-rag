import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# 1. Generate "Bulky" Mock Enterprise Documents
# In a real scenario, you would use PyPDFLoader or Unstructured to read actual files.
documents = [
    {
        "content": """
        Engineering Architecture v3.0: 
        The transition to microservices will be completed by Q4. The core transaction engine is moving from a monolithic PostgreSQL instance to distributed Cassandra nodes to handle high write-throughput. 
        API Gateway will be managed by Kong. All asynchronous messaging will shift from RabbitMQ to Apache Kafka. 
        Security Protocol: All internal microservice communication must enforce mTLS.
        """,
        "metadata": {"department": "Engineering", "clearance_level": 3, "source": "arch_v3.txt"}
    },
    {
        "content": """
        Project Titan - Executive Summary:
        Project Titan encompasses the aggressive acquisition strategy of our primary competitor, Apex Dynamics. 
        Valuation is currently estimated at $4.2 Billion. The merger will result in a 15% workforce reduction primarily in redundant HR and Marketing departments. 
        Target acquisition date is November 15th. Do not discuss outside of the C-suite.
        """,
        "metadata": {"department": "Executive", "clearance_level": 5, "source": "titan_memo.pdf"}
    },
    {
        "content": """
        HR Employee Handbook 2026:
        Remote Work Policy: Employees are allowed to work remotely 2 days a week (Tuesday and Thursday). 
        Health Benefits: The new provider is BlueCross. Dental and vision are fully covered for Tier 1 and Tier 2 employees.
        Leave Policy: Standard 20 days PTO. Bereavement leave has been extended to 5 days.
        """,
        "metadata": {"department": "HR", "clearance_level": 1, "source": "handbook_26.pdf"}
    }
]

# 2. Initialize the Text Splitter
# chunk_size: Maximum characters per chunk
# chunk_overlap: Characters shared between chunks to preserve context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150, 
    chunk_overlap=30,
    separators=["\n\n", "\n", ".", " "]
)

print("Splitting documents into chunks...")
chunked_data = []

for doc in documents:
    # Split the large text into smaller chunks
    chunks = text_splitter.split_text(doc["content"])
    
    for chunk in chunks:
        # Attach the exact same metadata to EVERY chunk from that document
        chunked_data.append({
            "text": chunk,
            "metadata": doc["metadata"]
        })

print(f"Split {len(documents)} documents into {len(chunked_data)} chunks.")

# 3. Setup Qdrant and Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(path="./qdrant_local_data")
collection_name = "enterprise_docs_v2"

# Recreate collection for a fresh start
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 4. Vectorize and Upload
points = []
print("Generating vectors and uploading to Qdrant...")

for item in chunked_data:
    vector = model.encode(item["text"]).tolist()
    payload = item["metadata"]
    payload["page_content"] = item["text"] # Store text for the LLM
    
    points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

client.upsert(collection_name=collection_name, points=points)
print("Data pipeline executed successfully!")
client.close()