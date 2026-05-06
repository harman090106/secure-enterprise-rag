import streamlit as st
# Import your existing pipeline function from your secure_rag.py file
# Make sure your secure_rag.py is in the same folder and the collection name matches!
from secure_rag import process_query 

st.set_page_config(page_title="Secure Enterprise RAG", page_icon="🔐")

st.title("🔐 Secure Enterprise AI Assistant")
st.markdown("This chatbot uses **Role-Based Access Control (RBAC)** at the vector database level. Change your user profile in the sidebar to see how data access changes.")

# --- SIDEBAR: User Authentication Simulation ---
with st.sidebar:
    st.header("User Profile Settings")
    st.markdown("Simulate different employee logins.")
    
    selected_dept = st.selectbox(
        "Department",
        ["Engineering", "HR", "Sales", "Executive", "All"]
    )
    
    selected_clearance = st.slider(
        "Clearance Level",
        min_value=1, max_value=5, value=1
    )
    
    st.info("Try asking about 'Project Titan' as an HR employee (Level 1) vs. an Executive (Level 5).")

# Update the active user profile based on sidebar
active_user = {"department": selected_dept, "clearance_level": selected_clearance}

# --- CHAT INTERFACE ---
# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about internal company data..."):
    
    # 1. Display user message in UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 3. Format chat history for the LangChain prompt
    # We only pass the last 4 messages to prevent context overflow
    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-4:]]
    )
    
    # 4. Generate assistant response using your Secure RAG logic
    with st.chat_message("assistant"):
        with st.spinner("Searching secure database..."):
            
            # Call the function we built in secure_rag.py
            response = process_query({
                "question": prompt,
                "user_profile": active_user,
                "chat_history": formatted_history
            })
            
            st.markdown(response)
            
    # 5. Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})