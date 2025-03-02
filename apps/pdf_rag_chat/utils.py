import os
import uuid
import json
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


# ---------------------------
# Pinecone Initialization
# ---------------------------
def initialize_pinecone(index_name):
    """
    Initialize connection to Pinecone and get the specified index.
    If the index doesn't exist, return None.

    Args:
        index_name (str): Name of the Pinecone index to connect to

    Returns:
        tuple: (pinecone_index, dimension) - The Pinecone index object and its dimension,
               or (None, None) if the index doesn't exist or can't be connected to
    """
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("Pinecone API key not found in environment variables.")
            return None, None

        pc = Pinecone(api_key=api_key)

        # Check if the index exists
        existing_indexes = pc.list_indexes()
        index_exists = False

        for idx in existing_indexes:
            # Depending on your Pinecone version, idx can be a dict or a string.
            name = idx.name if hasattr(idx, "name") else idx
            if name == index_name:
                index_exists = True
                break

        if not index_exists:
            print(f"Index '{index_name}' does not exist.")
            return None, None

        # Connect to the existing index
        index = pc.Index(index_name)

        # Get the dimension from the index description
        description = pc.describe_index(index_name)
        dimension = description.dimension

        print(
            f"Successfully connected to Pinecone index '{index_name}' with dimension {dimension}"
        )
        return index, dimension
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None, None


# ---------------------------
# PDF Processing Functions
# ---------------------------
def extract_text_from_pdf(pdf_file_path: str) -> str:
    try:
        reader = PdfReader(pdf_file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"Warning: Error extracting text from page {page_num}: {str(e)}")

        if not text.strip():
            print(
                "Warning: No text extracted from PDF. The PDF may be scanned or image-based."
            )

        return text
    except Exception as e:
        print(f"Error in extract_text_from_pdf: {str(e)}")
        raise ValueError(f"Failed to read PDF: {str(e)}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and (end - start) == chunk_size:
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
            elif last_newline > start + chunk_size // 2:
                end = last_newline + 1
        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length
    return chunks


# ---------------------------
# Embedding Functions
# ---------------------------
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-roberta-large-v1"
        )
    return _embedding_model


def embed_chunks(chunks):
    model = get_embedding_model()
    embeddings = []
    for chunk in chunks:
        embed = model.embed_documents([chunk])[0]
        embeddings.append(embed)
    return embeddings


def store_embeddings(index, chunks, embeddings, pdf_name: str):
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        i_end = min(i + batch_size, len(chunks))
        ids = [f"{pdf_name}-{uuid.uuid4()}" for _ in range(i, i_end)]
        metadata = [
            {"text": chunks[j], "pdf_name": pdf_name, "chunk_id": j}
            for j in range(i, i_end)
        ]
        vectors = [
            (ids[j - i], embeddings[j], metadata[j - i]) for j in range(i, i_end)
        ]
        index.upsert(vectors=vectors)
    return True


def search_similar_chunks(query: str, index, pdf_name: str = None, top_k: int = 5):
    model = get_embedding_model()
    query_embedding = model.embed_query(query)
    filter_query = {"pdf_name": pdf_name} if pdf_name else None
    results = index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter_query
    )
    return results.get("matches", [])


# ---------------------------
# Gemini (Google Generative AI) Functions
# ---------------------------
def initialize_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found in environment variables.")
    genai.configure(api_key=api_key)
    return True


def get_gemini_response(prompt: str, context: str = None, temperature: float = 0.7):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        if context:
            full_prompt = f"""
Context information:
{context}

Question: {prompt}

Please provide a helpful, accurate response based on the context. If the answer cannot be determined from the context, state that clearly.
"""
        else:
            full_prompt = prompt
        response = model.generate_content(
            full_prompt, generation_config={"temperature": temperature}
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


# ---------------------------
# Quiz and Assignment Generation Functions
# ---------------------------
def extract_pdf_subject(pdf_content: str) -> str:
    """
    Use Gemini to extract the main subject or topic from the PDF content.
    """
    subject_prompt = f"""
Extract the main subject or topic of the following PDF content in one clear sentence:
{pdf_content[:2000]}
"""
    subject_response = get_gemini_response(subject_prompt, temperature=0.2)
    subject = subject_response.strip()
    return subject


def generate_quiz(index, pdf_name: str, pdf_content: str, num_questions: int = 5):
    """
    Generate a quiz
    """
    subject = extract_pdf_subject(pdf_content)

    # Retrieve context from Pinecone using the subject as a query
    similar_chunks = search_similar_chunks(subject, index, pdf_name=pdf_name, top_k=10)
    context = "\n\n".join([match["metadata"]["text"] for match in similar_chunks])

    prompt = f"""
The about "{subject}". Using only the provided context, generate a quiz with {num_questions} multiple-choice questions that assess understanding of this subject. Every question must be directly based on the context. For each question, provide 4 options and clearly indicate the correct answer with its corresponding letter (A, B, C, or D). Include a brief explanation for why the correct answer is correct, based solely on the provided context.

Return only a JSON array formatted exactly as follows (do not include any additional text):
[
    {{
        "question": "Question text",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": "A",
        "explanation": "Brief explanation based on the context"
    }},
    ...
]

Context:
{context}
"""
    response = get_gemini_response(prompt, temperature=0.2)
    try:
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        else:
            json_str = response.strip()
        quiz_data = json.loads(json_str)
        return quiz_data
    except Exception as e:
        print("Error parsing quiz response:", e)
        return []


def generate_assignment(
    index,
    pdf_name: str,
    pdf_content: str,
    assignment_type: str = "short_answer",
    num_questions: int = 3,
):
    """
    Generate an assignment based on the ingested PDF context from the vector database.
    The function extracts the subject, retrieves similar chunks from Pinecone,
    and uses that context to instruct Gemini to generate assignment questions.
    """
    subject = extract_pdf_subject(pdf_content)

    # Retrieve context from Pinecone using the subject as a query
    similar_chunks = search_similar_chunks(subject, index, pdf_name=pdf_name, top_k=10)
    context = "\n\n".join([match["metadata"]["text"] for match in similar_chunks])

    prompt = f"""
The following context is about "{subject}". Based solely on the provided context, generate a {assignment_type} assignment with {num_questions} questions. If the assignment type is 'short_answer', create questions that require brief explanations; if 'essay', create deeper questions; if 'research', encourage further exploration of the topics. Use only the provided context to derive questions.

Return only a JSON array formatted exactly as follows (do not include any additional text):
[
    {{
        "question": "Question text",
        "hints": ["Hint 1", "Hint 2"],
        "key_points": ["Key point 1", "Key point 2", "Key point 3"]
    }},
    ...
]

Context:
{context}
"""
    response = get_gemini_response(prompt, temperature=0.3)
    try:
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        else:
            json_str = response.strip()
        assignment_data = json.loads(json_str)
        return assignment_data
    except Exception as e:
        print("Error parsing assignment response:", e)
        return []
