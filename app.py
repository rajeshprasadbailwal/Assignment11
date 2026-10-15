import os
import tempfile
import shutil
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from langserve import add_routes
from pydantic import BaseModel, Field

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# Google embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


# ============================================================
# 2. INITIAL KNOWLEDGE BASE
# ============================================================

default_doc = [
    Document(
        page_content=(
            "This is the default Knowledge Transfer document. "
            "Upload a technical PDF to provide project-specific "
            "documentation for the RAG assistant."
        )
    )
]

default_chunks = text_splitter.split_documents(default_doc)

# Build the initial FAISS vector store.
# This requires a working Google embedding API.
vector_store = FAISS.from_documents(
    default_chunks,
    embeddings,
)


# ============================================================
# 3. RAG QUESTION-ANSWERING LOGIC
# ============================================================

def answer_kt_question(question: str) -> str:
    """
    Retrieve relevant document chunks and answer
    the developer's question using Gemini.
    """
    global vector_store

    question = question.strip()

    if not question:
        return "Please enter a question."

    # Retrieve relevant documents
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    relevant_docs: List[Document] = retriever.invoke(question)

    if not relevant_docs:
        return (
            "I could not find relevant information "
            "in the uploaded documentation."
        )

    context_text = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert Technical Knowledge Transfer Assistant.

Answer developer questions using the supplied documentation.
Be clear, accurate, and helpful.

Rules:
1. Prioritize information from the provided context.
2. Do not invent project-specific details.
3. If the answer is not in the context, clearly say so.
4. Explain technical concepts in simple language when useful.
5. Use code examples when they help answer the question.

CONTEXT FROM DOCUMENTATION:
{context}

DEVELOPER QUESTION:
{question}

ANSWER:
"""
    )

    # Run the RAG chain
    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context_text,
            "question": question,
        }
    )

    return str(response.content)


# ============================================================
# 4. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RAG Knowledge Transfer Assistant",
    description=(
        "Upload technical PDF documents and ask "
        "questions about their contents."
    ),
    version="1.0.0",
)


# Input schema for LangServe
class RAGInput(BaseModel):
    question: str = Field(
        description=(
            "Your question about the uploaded "
            "technical documentation."
        )
    )


def run_rag_agent(inputs) -> str:
    """
    Accept the question from LangServe
    and call the RAG function.
    """
    if isinstance(inputs, dict):
        query = inputs["question"]
    else:
        query = inputs.question

    return answer_kt_question(query)


formatted_rag_chain = RunnableLambda(
    run_rag_agent
).with_types(
    input_type=RAGInput,
    output_type=str,
)


# LangServe endpoint:
# POST /agent/invoke
add_routes(
    app,
    formatted_rag_chain,
    path="/agent",
    playground_type="default",
)


# ============================================================
# 5. PDF UPLOAD AND INDEXING
# ============================================================

@app.post("/upload-kt-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its text, split it into chunks,
    and rebuild the FAISS vector store.
    """
    global vector_store

    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temp_path = None

    try:
        # Save the uploaded PDF to a temporary file.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_path = temp_file.name

            shutil.copyfileobj(file.file, temp_file)

        # Extract text from the PDF.
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF contains no pages.",
            )

        # Split the extracted text into chunks.
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The PDF contains no readable text. "
                    "Please upload a text-based PDF."
                ),
            )

        # Build a new index before replacing the old one.
        # If indexing fails, the existing store remains intact.
        new_vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        vector_store = new_vector_store

        return {
            "message": (
                "PDF successfully indexed and is ready "
                "for Knowledge Transfer questions!"
            ),
            "filename": filename,
            "total_pages": len(docs),
            "total_chunks": len(chunks),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"PDF indexing error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process or index the PDF. "
                "Check the Render logs for details."
            ),
        ) from exc

    finally:
        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 6. EMBEDDED FRONTEND
# ============================================================

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >
    <title>KT Knowledge Base Assistant</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, sans-serif;
        }

        body {
            background: #f4f6f8;
            color: #333;
            display: flex;
            justify-content: center;
            padding: 40px 20px;
            min-height: 100vh;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            padding: 32px;
        }

        h1 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 12px;
            text-align: center;
        }

        .subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 28px;
        }

        .card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }

        .card h2 {
            font-size: 1.1rem;
            color: #475569;
            margin-bottom: 12px;
        }

        .form-group {
            display: flex;
            gap: 12px;
        }

        input[type="file"] {
            flex: 1;
            min-width: 0;
            padding: 8px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            background: #fff;
        }

        input[type="text"] {
            flex: 1;
            min-width: 0;
            padding: 12px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            font-size: 0.95rem;
        }

        button {
            background: #2563eb;
            color: #fff;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s ease;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            background: #94a3b8;
            cursor: not-allowed;
        }

        .status {
            margin-top: 12px;
            font-size: 0.9rem;
            font-weight: 500;
            overflow-wrap: anywhere;
        }

        .status.success {
            color: #16a34a;
        }

        .status.error {
            color: #dc2626;
        }

        .answer-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        @media (max-width: 600px) {
            body {
                padding: 20px 12px;
            }

            .container {
                padding: 20px;
            }

            .form-group {
                flex-direction: column;
            }

            button {
                width: 100%;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>RAG Knowledge Transfer Assistant</h1>

        <p class="subtitle">
            Upload technical documentation and get answers
            to your development questions.
        </p>

        <div class="card">
            <h2>1. Upload Technical Document (PDF)</h2>

            <div class="form-group">
                <input
                    type="file"
                    id="pdfFile"
                    accept=".pdf,application/pdf"
                >

                <button id="uploadBtn" onclick="uploadPDF()">
                    Upload &amp; Index
                </button>
            </div>

            <div id="uploadStatus" class="status"></div>
        </div>

        <div class="card">
            <h2>2. Ask a Question</h2>

            <div class="form-group">
                <input
                    type="text"
                    id="questionInput"
                    placeholder="e.g., How does the API work?"
                >

                <button id="askBtn" onclick="askQuestion()">
                    Submit
                </button>
            </div>

            <div
                id="answerResult"
                class="answer-box"
                style="display: none;"
            ></div>
        </div>
    </div>

    <script>
        async function uploadPDF() {
            const fileInput = document.getElementById("pdfFile");
            const statusDiv = document.getElementById("uploadStatus");
            const uploadBtn = document.getElementById("uploadBtn");

            if (!fileInput.files.length) {
                statusDiv.className = "status error";
                statusDiv.innerText = "Please select a PDF file first.";
                return;
            }

            const selectedFile = fileInput.files[0];

            if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
                statusDiv.className = "status error";
                statusDiv.innerText = "Please select a PDF file.";
                return;
            }

            const formData = new FormData();
            formData.append("file", selectedFile);

            uploadBtn.disabled = true;
            statusDiv.className = "status";
            statusDiv.innerText = "Uploading and indexing PDF...";

            try {
                const response = await fetch("/upload-kt-pdf/", {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    statusDiv.className = "status success";
                    statusDiv.innerText =
                        result.message +
                        " Pages: " + result.total_pages +
                        ". Chunks: " + result.total_chunks + ".";
                } else {
                    statusDiv.className = "status error";
                    statusDiv.innerText =
                        result.detail || "Failed to index PDF.";
                }
            } catch (error) {
                statusDiv.className = "status error";
                statusDiv.innerText =
                    "Upload failed. Check your internet connection and server logs.";
            } finally {
                uploadBtn.disabled = false;
            }
        }

        async function askQuestion() {
            const questionInput =
                document.getElementById("questionInput");

            const answerDiv =
                document.getElementById("answerResult");

            const askBtn =
                document.getElementById("askBtn");

            const query = questionInput.value.trim();

            if (!query) {
                answerDiv.style.display = "block";
                answerDiv.innerText = "Please enter a question.";
                return;
            }

            askBtn.disabled = true;
            answerDiv.style.display = "block";
            answerDiv.innerText =
                "Searching documentation and generating an answer...";

            try {
                const response = await fetch("/agent/invoke", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        input: {
                            question: query
                        }
                    })
                });

                const result = await response.json();

                if (response.ok) {
                    answerDiv.innerText =
                        typeof result.output === "string"
                            ? result.output
                            : JSON.stringify(result.output, null, 2);
                } else {
                    answerDiv.innerText =
                        "Error: " +
                        (result.detail || JSON.stringify(result));
                }
            } catch (error) {
                answerDiv.innerText =
                    "Unable to connect to the assistant API. Please try again.";
            } finally {
                askBtn.disabled = false;
            }
        }

        document.getElementById("questionInput")
            .addEventListener("keydown", function(event) {
                if (event.key === "Enter") {
                    askQuestion();
                }
            });
    </script>
</body>
</html>
"""


# ============================================================
# 7. FRONTEND AND HEALTH ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=HTML_UI)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "RAG Knowledge Transfer Assistant",
    }


# ============================================================
# 8. SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
