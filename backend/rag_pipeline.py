import os
from dotenv import load_dotenv
from groq import Groq
from backend.pdf_parser import extract_text_from_pdf, split_text_into_chunks
from backend.embeddings import create_embeddings, create_faiss_index, search_vectors

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.index = None
        self.chunks = []

    def process_pdf(self, pdf_path: str):
        """
        1. Extract text from PDF
        2. Split into chunks
        3. Create embeddings and FAISS index
        """
        print(f"Processing PDF: {pdf_path}")
        
        # 1. Extract Text
        raw_text = extract_text_from_pdf(pdf_path)
        
        # 2. Chunk Text
        self.chunks = split_text_into_chunks(raw_text, chunk_size=500, overlap=50)
        
        if not self.chunks:
            raise ValueError("No text extracted from PDF.")

        # 3. Create Embeddings & Index
        embeddings = create_embeddings(self.chunks)
        self.index = create_faiss_index(embeddings)
        print(f"Indexed {len(self.chunks)} chunks.")

    def generate_answer(self, query: str) -> dict:
        """
        1. Search for relevant chunks
        2. Construct prompt with context
        3. Call Groq API
        4. Return answer and source citations
        """
        if self.index is None:
            return {"answer": "Please upload a PDF first.", "sources": []}

        # 1. Retrieve relevant chunks
        indices = search_vectors(self.index, query, k=3)
        relevant_chunks = [self.chunks[i] for i in indices if i < len(self.chunks)]
        
        # 2. Construct Context
        context = "\n\n".join(relevant_chunks)
        
        # 3. Prompt Engineering
        prompt = f"""
        You are a helpful AI assistant. Answer the question based ONLY on the following context.
        If the answer is not in the context, say "I cannot find the answer in the provided document."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """

        # 4. Call Groq API (Llama 3)
        completion = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        answer = completion.choices[0].message.content
        
        # 5. Prepare Sources (for UI display)
        sources = [{"chunk_index": idx, "preview": chunk[:100] + "..."} for idx, chunk in zip(indices, relevant_chunks)]

        return {"answer": answer, "sources": sources}