import streamlit as st
import requests
import time

# Configuration
BACKEND_URL = "http://localhost:8000"  # Change this when deploying to Render

st.set_page_config(page_title="Smart PDF Assistant", page_icon="📄")

st.title("📄 Smart PDF Assistant")
st.markdown("Upload a PDF and ask questions to get AI-powered answers with source citations.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for File Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Send file to backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{BACKEND_URL}/upload-pdf", files=files)
                if response.status_code == 200:
                    st.success("PDF processed successfully!")
                    # Clear chat history on new upload
                    st.session_state.messages = []
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if it's an AI response
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1} (Chunk {source['chunk_index']}):**")
                    st.caption(source["preview"])

# User Input
if prompt := st.chat_input("Ask a question about the PDF..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask", 
                    json={"question": prompt}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Display sources in expander
                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1} (Chunk {source['chunk_index']}):**")
                                st.caption(source["preview"])
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Connection Error: Is the backend running? {str(e)}")