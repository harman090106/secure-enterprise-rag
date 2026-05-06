import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load the Groq API Key from your .env file
load_dotenv()

# 2. Setup the Embedding Model & Connect to Qdrant
print("Connecting to database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(path="./qdrant_local_data")

# This is LangChain's wrapper that lets it talk to our Qdrant instance
vector_store = QdrantVectorStore(
    client=client, 
    collection_name="enterprise_docs", 
    embedding=embeddings
)

# 3. Create the Retriever
# "k=2" means it will fetch the 2 most relevant document chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. Initialize the Groq LLM
# We are using Llama 3 (8 billion parameters) hosted on Groq for blazing fast speed
print("Initializing LLM...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

# 5. Define the Prompt Template
# This is how we instruct the LLM to behave
template = """You are a highly capable internal company assistant. 
Use the following retrieved context to answer the user's question. 
If you don't know the answer or the context doesn't contain it, just say "I don't have access to that information." Do not make up an answer.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Build the LCEL Chain (LangChain Expression Language)
# This is the modern way to build LangChain pipelines
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Test the Pipeline
if __name__ == "__main__":
    print("\n--- Testing Pipeline ---")
    
    # Try changing this question to ask about the Sales strategy or Project Titan!
    # test_question = "What is our company's new hybrid work policy?"
    test_question = "Tell me about project Titan"
    
    print(f"User: {test_question}")
    print("Generating answer...")
    
    response = rag_chain.invoke(test_question)
    
    print(f"\nAssistant: {response}")

    client.close()