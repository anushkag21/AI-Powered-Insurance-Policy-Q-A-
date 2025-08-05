from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Union
from contextlib import asynccontextmanager
import aiohttp
import tempfile
import os
from data_extraction import DataExtractor
from chunks import Chunking
from vectordb import VectorDB
import datetime
from groq import Groq
import uvicorn
import time

# Global variables for reusing connections
groq_client = Groq(api_key="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")
vector_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    # Startup
    global vector_db
    vector_db = VectorDB()
    print("✅ RAG API Server started successfully!")
    yield
    # Shutdown
    print("🔄 RAG API Server shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG API Server", 
    description="Enhanced RAG API with multi-format support", 
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class DocumentRequest(BaseModel):
    documents: Union[HttpUrl, str]  # URL or file path
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG API Server is running",
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vector_db": "connected" if vector_db else "disconnected",
        "groq_client": "connected",
        "timestamp": datetime.datetime.now().isoformat()
    }

async def download_document(url: str) -> bytes:
    """Download document from URL"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to download document: HTTP {response.status}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading document: {str(e)}")

def process_document(file_content: bytes, filename: str = "document") -> str:
    """Process document and return extracted text"""
    start_time = time.time()
    
    # Save content to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name
    
    try:
        # Extract text using existing DataExtractor
        extractor = DataExtractor()
        documents = extractor.from_pdf(temp_path)
        full_text = "\n".join([doc.text for doc in documents])
        
        # Performance optimization: Clean text
        full_text = ' '.join(full_text.split())
        
        print(f"[Time] Document extraction: {time.time() - start_time:.2f} seconds")
        return full_text
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def create_vector_index(text: str) -> int:
    """Create vector index from text and return number of chunks"""
    start_time = time.time()
    
    # Chunk the text
    chunker = Chunking()
    refined_chunks = chunker.from_text(text)
    print(f"[Time] Text chunking: {time.time() - start_time:.2f} seconds")
    
    # Create and populate vector index
    index_start = time.time()
    vector_db.create_index()
    print(f"[Time] Index creation: {time.time() - index_start:.2f} seconds")
    
    upsert_start = time.time()
    vector_db.upsert_chunks(refined_chunks)
    print(f"[Time] Vector upsert: {time.time() - upsert_start:.2f} seconds")
    
    return len(refined_chunks)

async def process_question(question: str, question_idx: int) -> str:
    """Process a single question and return answer"""
    q_start = time.time()
    
    try:
        # Step 1: Rewrite query for better retrieval
        rewrite_prompt = (
            "Rewrite the following question to be more specific and include relevant context for document search. "
            "If the question is already clear, keep it unchanged. Keep it concise.\n\n"
            f"Original question: {question}\nRewritten question:"
        )
        
        rewrite_completion = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.2,
            max_completion_tokens=128,
            top_p=1,
            stream=False,
            stop=None,
            timeout=30
        )
        rewritten_query = rewrite_completion.choices[0].message.content.strip()
        print(f"[Time] Q{question_idx} query rewrite: {time.time() - q_start:.2f} seconds")

        # Step 2: Retrieve relevant chunks
        retrieval_start = time.time()
        pinecone_results = vector_db.query(rewritten_query, top_k=3)
        print(f"[Time] Q{question_idx} vector search: {time.time() - retrieval_start:.2f} seconds")

        # Step 3: Build context from results
        context = ""
        if 'result' in pinecone_results and 'hits' in pinecone_results['result']:
            for i, hit in enumerate(pinecone_results['result']['hits']):
                chunk_text = hit.get('fields', {}).get('chunk_text', 'No text available')
                # Optimize: Truncate long chunks
                if len(chunk_text) > 1000:
                    chunk_text = chunk_text[:1000] + "..."
                context += f"Source {i+1}: {chunk_text}\n\n"
        
        if not context.strip():
            return "I couldn't find relevant information in the document to answer this question."

        # Step 4: Generate answer
        llm_start = time.time()
        answer_prompt = (
            f"Based on the following information from a document, provide a clear and concise answer to this question: {question}\n\n"
            f"Document Information:\n{context}\n\n"
            "Instructions:\n"
            "- Answer directly and concisely\n"
            "- Use only information from the provided sources\n"
            "- If the information is not available, state that clearly\n"
            "- Keep the answer under 200 words\n\n"
            "Answer:"
        )
        
        completion = groq_client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.3,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
            timeout=45
        )
        
        answer = completion.choices[0].message.content.strip()
        print(f"[Time] Q{question_idx} answer generation: {time.time() - llm_start:.2f} seconds")
        print(f"[Time] Q{question_idx} total: {time.time() - q_start:.2f} seconds")
        
        return answer
        
    except Exception as e:
        print(f"Error processing question {question_idx}: {str(e)}")
        return f"Error processing question: {str(e)}"

@app.post("/hackrx/run", response_model=AnswerResponse)
async def process_documents_and_questions(request: DocumentRequest):
    """
    Main API endpoint to process documents and answer questions
    Expects JSON: {"documents": "url_or_path", "questions": ["q1", "q2", ...]}
    Returns JSON: {"answers": ["a1", "a2", ...]}
    """
    total_start = time.time()
    
    try:
        # Step 1: Download/Load document
        print(f"📄 Processing document: {request.documents}")
        
        if str(request.documents).startswith(('http://', 'https://')):
            # Download from URL
            file_content = await download_document(str(request.documents))
        else:
            # Read from local file
            try:
                with open(str(request.documents), 'rb') as f:
                    file_content = f.read()
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"File not found: {request.documents}")
        
        # Step 2: Extract text from document
        print("🔍 Extracting text from document...")
        document_text = process_document(file_content)
        
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
        # Step 3: Create vector index
        print("🚀 Creating vector index...")
        num_chunks = create_vector_index(document_text)
        print(f"✅ Successfully indexed {num_chunks} chunks")
        
        # Step 4: Process all questions
        print(f"❓ Processing {len(request.questions)} questions...")
        answers = []
        
        for idx, question in enumerate(request.questions, 1):
            print(f"\n🤔 Processing Question {idx}: {question[:100]}...")
            answer = await process_question(question, idx)
            answers.append(answer)
        
        print(f"\n🎉 Successfully processed all questions!")
        print(f"⏱️  Total processing time: {time.time() - total_start:.2f} seconds")
        
        return AnswerResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in main processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# File upload endpoint removed to avoid multipart dependency

if __name__ == "__main__":
    print("🚀 Starting RAG API Server...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    print("🔗 Main endpoint: POST http://localhost:8000/hackrx/run")
    print("💡 Health check: GET http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        access_log=True
    )