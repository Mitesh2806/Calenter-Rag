from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
from apps.pdf_rag_chat.utils import (
    extract_text_from_pdf,
    chunk_text,
    embed_chunks,
    store_embeddings,
    initialize_pinecone,
    search_similar_chunks,
    initialize_gemini,
    get_gemini_response,
    generate_quiz,
    generate_assignment
)
import pinecone
from pinecone import Pinecone, ServerlessSpec

router = APIRouter()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "index1")
# Try to initialize Pinecone index; if not found, pinecone_index will be None.
pinecone_index, index_dimensions = initialize_pinecone(index_name=INDEX_NAME)

# Initialize Gemini
try:
    initialize_gemini()
except Exception as e:
    raise RuntimeError(f"Error initializing Gemini: {e}")

# ---------------------------
# PDF Upload Endpoint
# ---------------------------
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global pinecone_index, index_dimensions
    suffix = os.path.splitext(file.filename)[1]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process PDF: extract text, chunk, embed
        pdf_text = extract_text_from_pdf(tmp_path)
        os.remove(tmp_path)
        chunks = chunk_text(pdf_text)
        embeddings = embed_chunks(chunks)
        
        # If the Pinecone index isn't initialized, create or connect to it
        if pinecone_index is None:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            dimension = int(os.getenv("INDEX_DIMENSION", "1024"))
            
            # Check if the index already exists before trying to create it
            try:
                existing_indexes = pc.list_indexes()
                index_exists = False
                for index in existing_indexes:
                    name = index.name if hasattr(index, "name") else index
                    if name == INDEX_NAME:
                        index_exists = True
                        break
                if not index_exists:
                    pc.create_index(
                        name=INDEX_NAME,
                        dimension=dimension,
                        metric='cosine',
                        spec=ServerlessSpec(cloud='aws', region=os.getenv("PINECONE_REGION", "us-east-1"))
                    )
            except Exception as e:
                if "ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
                    pass
                else:
                    raise e
            
            # Reinitialize to get the index (whether it was newly created or already existed)
            pinecone_index, index_dimensions = initialize_pinecone(index_name=INDEX_NAME)
            if pinecone_index is None:
                raise HTTPException(status_code=500, detail="Failed to connect to Pinecone index after creation/verification.")
        
        store_embeddings(pinecone_index, chunks, embeddings, pdf_name=file.filename)
        return JSONResponse(content={"message": f"PDF '{file.filename}' processed and embeddings stored."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Chat Endpoint
# ---------------------------
@router.post("/chat")
async def chat(query: str = Form(...), pdf_name: str = Form(None)):
    try:
        if pinecone_index is None:
            raise HTTPException(status_code=500, detail="Pinecone index is not initialized.")
        matches = search_similar_chunks(query, index=pinecone_index, pdf_name=pdf_name)
        context = "\n\n".join([match["metadata"]["text"] for match in matches])
        response_text = get_gemini_response(query, context=context)
        return JSONResponse(content={"response": response_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Quiz and Assignment Request Models
# ---------------------------
class QuizRequest(BaseModel):
    pdf_name: str        # Added pdf_name field
    pdf_content: str
    num_questions: int = 5

class AssignmentRequest(BaseModel):
    pdf_name: str        # Added pdf_name field
    pdf_content: str
    assignment_type: str = "short_answer"
    num_questions: int = 3

# ---------------------------
# Quiz Generation Endpoint
# ---------------------------
@router.post("/quiz")
async def generate_quiz_endpoint(request: QuizRequest):
    try:
        # Pass pinecone_index, pdf_name, and pdf_content to generate_quiz
        quiz_data = generate_quiz(pinecone_index, request.pdf_name, request.pdf_content, num_questions=request.num_questions)
        return JSONResponse(content={"quiz": quiz_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Assignment Generation Endpoint
# ---------------------------
@router.post("/assignment")
async def generate_assignment_endpoint(request: AssignmentRequest):
    try:
        # Pass pinecone_index, pdf_name, and pdf_content to generate_assignment
        assignment_data = generate_assignment(
            pinecone_index,
            request.pdf_name,
            request.pdf_content,
            assignment_type=request.assignment_type,
            num_questions=request.num_questions
        )
        return JSONResponse(content={"assignment": assignment_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
