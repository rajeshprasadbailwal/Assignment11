import os
import shutil
import tempfile
import uvicorn
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from langserve import add_routes
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# 1. API KEY & MODEL SETUP
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# LLM setup using standard Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Text embeddings setup
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

# Initialize global text splitter and vector store fallback
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

default_doc = [
    Document(
        page_content="This is the default Knowledge Transfer document. Upload a PDF using the frontend form to replace this data."
    )
]
default_chunks = text_splitter.split_documents(default_doc)
vector_store = FAISS.from_documents(default_chunks, embeddings)


# ============================================================
# 2. RAG LOGIC
# ============================================================

def answer_kt_question(question: str) -> str:
    global vector_store

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs: List[Document] = retriever.invoke(question)

    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_template(
        """You are an expert Technical Knowledge Transfer (KT) Assistant.
Provide accurate, source-grounded answers based strictly on the provided context.

CONTEXT FROM DOCUMENTATION:
{context}

DEVELOPER QUESTION:
{question}
"""
    )

    chain = prompt | llm
    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    return response.content


# ============================================================
# 3. FASTAPI APP & ENDPOINTS
# ============================================================

app = FastAPI(
    title="RAG Knowledge Transfer (KT) Agent",
    version="1.0"
)

class RAGInput(BaseModel):
    question: str = Field(description="Your question regarding technical setup or architecture")

def run_rag_agent(inputs) -> str:
    query = inputs["question"] if isinstance(inputs, dict) else inputs.question
    return answer_kt_question(query)

formatted_rag_chain = RunnableLambda(run_rag_agent).with_types(
    input_type=RAGInput, 
    output_type=str
)

# LangServe API routes
add_routes(
    app,
    formatted_rag_chain,
    path="/agent",
    playground_type="default"
)


# Upload endpoint for indexing PDFs
@app.post("/upload-kt-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Secure temporary file handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        if not docs:
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no readable text.")

        chunks = text_splitter.split_documents(docs)
        
        # Atomically rebuild vector store index
        new_vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store = new_vector_store

        return {
            "message": "PDF successfully indexed and ready for KT queries!",
            "filename": file.filename,
            "total_chunks": len(chunks)
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 4. EMBEDDED FRONTEND UI
# ============================================================

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KT Knowledge Base Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }
        body { background: #f4f6f8; color: #333; display: flex; justify-content: center; padding: 40px 20px; }
        .container { width: 100%; max-width: 800px; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); padding: 32px; }
        h1 { font-size: 1.8rem; font-weight: 700; color: #1e293b; margin-bottom: 24px; text-align: center; }
        .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 24px; }
        .card h2 { font-size: 1.1rem; color: #475569; margin-bottom: 12px; }
        .form-group { display: flex; gap: 12px; }
        input[type="file"] { flex: 1; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; }
        input[type="text"] { flex: 1; padding: 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.95rem; }
        button { background: #2563eb; color: #fff; border: none; padding: 12px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s ease; }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #94a3b8; cursor: not-allowed; }
        .status { margin-top: 10px; font-size: 0.9rem; font-weight: 500; }
        .status.success { color: #16a34a; }
        .status.error { color: #dc2626; }
        .answer-box { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-top: 16px; white-space: pre-wrap; font-size: 0.95rem; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Knowledge Transfer Assistant</h1>

        <!-- PDF UPLOAD SECTION -->
        <div class="card">
            <h2>1. Upload Technical Document (PDF)</h2>
            <div class="form-group">
                <input type="file" id="pdfFile" accept=".pdf">
                <button id="uploadBtn" onclick="uploadPDF()">Upload & Index</button>
            </div>
            <div id="uploadStatus" class="status"></div>
        </div>

        <!-- QUERY SECTION -->
        <div class="card">
            <h2>2. Ask a Question</h2>
            <div class="form-group">
                <input type="text" id="questionInput" placeholder="e.g., How do I set up the API authentication?">
                <button id="askBtn" onclick="askQuestion()">Submit</button>
            </div>
            <div id="answerResult" class="answer-box" style="display: none;"></div>
        </div>
    </div>

    <script>
        async function uploadPDF() {
            const fileInput = document.getElementById('pdfFile');
            const statusDiv = document.getElementById('uploadStatus');
            const uploadBtn = document.getElementById('uploadBtn');

            if (!fileInput.files.length) {
                statusDiv.className = "status error";
                statusDiv.innerText = "Please select a PDF file first.";
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            uploadBtn.disabled = true;
            statusDiv.className = "status";
            statusDiv.innerText = "Indexing document... Please wait.";

            try {
                const response = await fetch('/upload-kt-pdf/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (response.ok) {
                    statusDiv.className = "status success";
                    statusDiv.innerText = `${result.message} (${result.total_chunks} chunks indexed)`;
                } else {
                    statusDiv.className = "status error";
                    statusDiv.innerText = result.detail || "Failed to index PDF.";
                }
            } catch (err) {
                statusDiv.className = "status error";
                statusDiv.innerText = "An error occurred during upload.";
            } finally {
                uploadBtn.disabled = false;
            }
        }

        async function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const answerDiv = document.getElementById('answerResult');
            const askBtn = document.getElementById('askBtn');
            const query = questionInput.value.trim();

            if (!query) return;

            askBtn.disabled = true;
            answerDiv.style.display = "block";
            answerDiv.innerText = "Searching documentation and generating answer...";

            try {
                const response = await fetch('/agent/invoke', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input: { question: query } })
                });

                const result = await response.json();

                if (response.ok) {
                    answerDiv.innerText = result.output;
                } else {
                    answerDiv.innerText = "Error: " + (result.detail || "Unable to get an answer.");
                }
            } catch (err) {
                answerDiv.innerText = "Network error while connecting to the assistant API.";
            } finally {
                askBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=HTML_UI)


# ============================================================
# 5. SERVER ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
