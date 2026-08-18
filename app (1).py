import os
import uvicorn
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from langserve import add_routes
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import shutil


# ============================================================
# 1. API KEY & MODEL SETUP
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# FIXED: Updated embedding model name to text-embedding-004
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Initialize vector store with a default fallback document so it never errors out
default_doc = [Document(page_content="This is the default Knowledge Transfer document for the RAG Agent. It covers project architecture, API setups, and onboarding guidelines.")]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
default_chunks = text_splitter.split_documents(default_doc)
vector_store = FAISS.from_documents(default_chunks, embeddings)


# ============================================================
# 2. RAG FUNCTIONS
# ============================================================

def answer_kt_question(question: str) -> str:
    global vector_store
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs: List[Document] = retriever.invoke(question)
    
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Technical Knowledge Transfer (KT) Assistant.
        Project knowledge is often scattered across large PDF documents such as architecture, setup, API, and technical documentation.
        Provide accurate, source-grounded answers to developers based strictly on the provided documents to reduce manual document searching and make onboarding faster.

        CONTEXT FROM PDF:
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
# 3. FASTAPI APP & ROUTES
# ============================================================

app = FastAPI(
    title="RAG Knowledge Transfer (KT) Agent",
    version="1.0"
)

class RAGInput(BaseModel):
    question: str = Field(description="Your question regarding the architecture, setup, or technical PDF")

def run_rag_agent(inputs) -> str:
    query = inputs["question"] if isinstance(inputs, dict) else inputs.question
    return answer_kt_question(query)

formatted_rag_chain = (
    RunnableLambda(run_rag_agent)
).with_types(input_type=RAGInput, output_type=str)

add_routes(
    app,
    formatted_rag_chain,
    path="/agent",
    playground_type="default"
)

@app.post("/upload-kt-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_file_path = f"./{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    
    # Overwrite global vector store with the newly uploaded PDF data
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return {
        "message": "PDF successfully indexed and ready for Knowledge Transfer queries!",
        "filename": file.filename,
        "total_chunks": len(chunks)
    }

@app.get("/")
def home():
    return {"message": "RAG Knowledge Transfer Agent is running perfectly!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
