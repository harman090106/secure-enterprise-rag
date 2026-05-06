import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model (downloads automatically the first time)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')



# 2. Connect to our local Qdrant database
client = QdrantClient(path="./qdrant_local_data")
collection_name = "enterprise_docs"

# 3. Define our Mock Enterprise Data with Metadata
# Notice the "metadata" dictionary - this is what we use for Access Control!
mock_documents = [
    {
        "text": "The new hybrid work policy requires employees to be in the office 3 days a week. Flexible hours apply.",
        "metadata": {"department": "HR", "clearance_level": 1, "doc_type": "policy"}
    },
    {
        "text": "The Q3 Sales strategy focuses on expanding into the European market, targeting enterprise software companies.",
        "metadata": {"department": "Sales", "clearance_level": 2, "doc_type": "strategy"}
    },
    {
        "text": "Backend architecture V2: The new microservices will use Node.js and communicate via a Kafka event broker.",
        "metadata": {"department": "Engineering", "clearance_level": 2, "doc_type": "technical"}
    },
    {
        "text": "Project 'Titan' involves a potential merger with our main competitor. This is strictly confidential.",
        "metadata": {"department": "Executive", "clearance_level": 5, "doc_type": "confidential"}
    },
    {
        "text": "General Company Info: Our cafeteria opens at 8 AM and serves free coffee all day.",
        "metadata": {"department": "All", "clearance_level": 1, "doc_type": "general"}
    }
]

# 4. Process and upload the data
points = []
print("Generating vectors and preparing data for upload...")

for doc in mock_documents:
    # Convert the text into a vector (a list of 384 numbers)
    vector = model.encode(doc["text"]).tolist()
    
    # We store the original text inside the metadata payload so the LLM can read it later
    payload = doc["metadata"]
    payload["page_content"] = doc["text"] 
    
    # Create a Qdrant Point (ID + Vector + Metadata)
    point = PointStruct(
        id=str(uuid.uuid4()), # Generate a unique random ID
        vector=vector,
        payload=payload
    )
    points.append(point)

# 5. Insert into the database
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Successfully ingested {len(points)} documents into Qdrant!")

# Always close the client when done to release the file lock
client.close()