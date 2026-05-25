import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(title="Smart PDF Assistant API")

# Enable CORS so Streamlit (frontend) can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline (Singleton pattern for this session)
rag_pipeline = RAGPipeline()

# Define Request/Response Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF, saves it temporarily, and processes it into the RAG pipeline.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Create a unique filename to avoid conflicts
    temp_filename = f"temp_{uuid.uuid4()}.pdf"
    
    try:
        # Save the file locally in Codespace
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF into the RAG pipeline
        rag_pipeline.process_pdf(temp_filename)
        
        return {"message": "PDF processed successfully", "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Takes a question and returns an answer based on the uploaded PDF.
    """
    try:
        result = rag_pipeline.generate_answer(request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))