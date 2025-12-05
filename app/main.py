from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PyPDF2 import PdfReader
from src.summarize import summarize_text
from src.ner import extract_entities
from src.qa import answer_question
from src.groq_qa import (
    answer_question_groq,
    summarize_with_groq,
    chat_qa_with_groq,
)
from src.rag import (
    answer_question_rag,
    summarize_rag,
    chunk_text,
    build_index,
    retrieve_top_k,
)
from src.risk_classifier import (
    classify_legal_risk,
    classify_legal_risk_sections,
    RISK_LABELS,
)
from typing import Literal

app = FastAPI(
    title="Legal Document Assistant API",
    description="Summarization for legal/policy documents.",
    version="0.1.0",
)

# Allowing frontend (React) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    text: str
    max_new_tokens: int | None = 256
    # Default "t5" so existing frontend keeps working.
    engine: Literal["t5", "bart"] = "t5"


class SummarizeResponse(BaseModel):
    summary: str

class SummarizeGenRequest(BaseModel):
    text: str
    max_new_tokens: int | None = 256


class SummarizeGenResponse(BaseModel):
    summary: str


class SummarizeRagRequest(BaseModel):
    text: str
    max_new_tokens: int | None = 256
    top_k: int | None = 5


class SummarizeRagResponse(BaseModel):
    summary: str
    retrieved_chunks: list[str]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(payload: SummarizeRequest):
    """
    Summarizing using fine-tuned models (T5 or BART).

    Default engine = "t5" (fine-tuned t5-billsum).
    Optionally, engine = "bart" to use fine-tuned BART.
    """
    summary = summarize_text(
        payload.text,
        max_new_tokens=payload.max_new_tokens or 256,
        engine=payload.engine,
    )
    return SummarizeResponse(summary=summary)



@app.post("/summarize_groq", response_model=SummarizeGenResponse)
def summarize_groq_endpoint(payload: SummarizeGenRequest):
    """
    Summarizing text using Groq LLM (Llama3).
    """
    summary = summarize_with_groq(
        payload.text,
        max_tokens=payload.max_new_tokens or 256,
    )
    return SummarizeGenResponse(summary=summary)

@app.post("/summarize_rag", response_model=SummarizeRagResponse)
def summarize_rag_endpoint(payload: SummarizeRagRequest):
    """
    RAG-based summarization:
    - Retrieve top-k chunks via embeddings
    - Summarize them with Groq
    """
    result = summarize_rag(
        full_text=payload.text,
        top_k=payload.top_k or 5,
    )
    return SummarizeRagResponse(
        summary=result["summary"],
        retrieved_chunks=result["retrieved_chunks"],
    )


class NerRequest(BaseModel):
    text: str

class NerResponse(BaseModel):
    entities: list

@app.post("/ner", response_model=NerResponse)
def ner_endpoint(payload: NerRequest):
    """
    Extract entities (ORG, PERSON, DATE, MONEY, LAW REFERENCES, etc.)
    from legal/policy text.
    """
    result = extract_entities(payload.text)
    return NerResponse(entities=result["entities"])

class QARequest(BaseModel):
    question: str
    context: str


class QAResponse(BaseModel):
    answer: str
    score: float
    start: int
    end: int

@app.post("/qa", response_model=QAResponse)
def qa_endpoint(payload: QARequest):
    """
    Answer a question given a legal/policy context.
    Uses extractive QA (span prediction).
    """
    result = answer_question(
        question=payload.question,
        context=payload.context,
    )

    return QAResponse(
        answer=result["answer"],
        score=result["score"],
        start=result["start"],
        end=result["end"],
    )

class QAResult(BaseModel):
    answer: str
    score: float
    start: int
    end: int

class RiskRequest(BaseModel):
    text: str


class RiskResponse(BaseModel):
    top_label: str
    scores: dict[str, float]

class SectionRisk(BaseModel):
    title: str
    text: str
    top_label: str
    scores: dict[str, float]


class SectionRiskResponse(BaseModel):
    sections: list[SectionRisk]


@app.post("/risk", response_model=RiskResponse)
def risk_endpoint(payload: RiskRequest):
    """
    Classify the legal risk level of the given text
    into Low / Medium / High using a transformer-based
    zero-shot classifier.
    """
    scores = classify_legal_risk(payload.text)

    # Here Picking label with highest score
    top_label = max(scores.items(), key=lambda x: x[1])[0]

    return RiskResponse(
        top_label=top_label,
        scores=scores,
    )

@app.post("/risk_sections", response_model=SectionRiskResponse)
def risk_sections_endpoint(payload: RiskRequest):
    """
    Analyze legal risk per section.

    Steps:
      - Split the document into sections (based on 'SECTION <number>' pattern).
      - Run risk classification (Low / Medium / High) on each section.
    """
    results = classify_legal_risk_sections(payload.text)
    return SectionRiskResponse(
        sections=[
            SectionRisk(
                title=r["title"],
                text=r["text"],
                top_label=r["top_label"],
                scores=r["scores"],
            )
            for r in results
        ]
    )


class AnalyzeRequest(BaseModel):
    text: str
    question: str | None = None      
    max_new_tokens: int | None = 256  


class AnalyzeResponse(BaseModel):
    summary: str
    entities: list
    qa: QAResult | None = None        

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(payload: AnalyzeRequest):
    """
    Combined endpoint:
    - Summarizes the input text
    - Extracts entities
    - Optionally answers a question about the text
    """
    # 1. Summary
    summary = summarize_text(
        text=payload.text,
        max_new_tokens=payload.max_new_tokens or 256,
    )

    # 2. NER
    ner_result = extract_entities(payload.text)
    entities = ner_result["entities"]

    # 3. Optional QA
    qa_result = None
    if payload.question:
        qa_raw = answer_question(
            question=payload.question,
            context=payload.text,
        )
        qa_result = QAResult(
            answer=qa_raw["answer"],
            score=qa_raw["score"],
            start=qa_raw["start"],
            end=qa_raw["end"],
        )

    return AnalyzeResponse(
        summary=summary,
        entities=entities,
        qa=qa_result,
    )

class QAGenRequest(BaseModel):
    question: str
    context: str
    max_new_tokens: int | None = None


class QAGenResponse(BaseModel):
    answer: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatQARequest(BaseModel):
    context: str
    messages: list[ChatMessage]


class ChatQAResponse(BaseModel):
    reply: str


@app.post("/qa_gen", response_model=QAGenResponse)
def qa_gen_endpoint(payload: QAGenRequest):
    """
    Generative QA endpoint (Groq Llama3):
    Returns a natural-language answer, not just a span.
    """
    answer = answer_question_groq(
        question=payload.question,
        context=payload.context,
    )
    return QAGenResponse(answer=answer)

@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from an uploaded PDF and return it as plain text.
    For now we only support .pdf files.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for text extraction.",
        )

    try:
        # Read file content into memory
        content = await file.read()
        pdf_bytes = io.BytesIO(content)

        reader = PdfReader(pdf_bytes)
        extracted_text_parts = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            extracted_text_parts.append(page_text)

        full_text = "\n\n".join(extracted_text_parts).strip()

        if not full_text:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in the PDF (might be scanned or image-based).",
            )

        return {"text": full_text}

    except HTTPException:
        # re-raise our own HTTPException
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from PDF: {e}",
        )

class QARagRequest(BaseModel):
  question: str
  context: str
  top_k: int | None = 3


class QARagResponse(BaseModel):
  answer: str
  retrieved_chunks: list[str]

@app.post("/qa_rag", response_model=QARagResponse)
def qa_rag_endpoint(payload: QARagRequest):
    """
    RAG-based QA:
    - Splits the full context into chunks
    - Retrieves top-k relevant chunks using embeddings
    - Asks Groq with only those chunks
    """
    result = answer_question_rag(
        question=payload.question,
        full_context=payload.context,
        top_k=payload.top_k or 3,
    )
    return QARagResponse(
        answer=result["answer"],
        retrieved_chunks=result["retrieved_chunks"],
    )

@app.post("/chat_qa", response_model=ChatQAResponse)
def chat_qa_endpoint(payload: ChatQARequest):
    """
    Chat-style QA over the current context using Groq Llama3 + RAG.

    Steps:
      1. Take the last user question from the chat history.
      2. Chunk + embed the full context.
      3. Retrieve top-k most relevant chunks for that question.
      4. Send ONLY those chunks (rag_context) + trimmed history to Groq.
    """
    history = payload.messages

    # 1) Get last user question (fallback if none)
    last_user_question = None
    for m in reversed(history):
        if m.role == "user":
            last_user_question = m.content
            break

    if not last_user_question:
        last_user_question = (
            "Answer the user's questions about the legal document as clearly as possible."
        )

    # 2) Chunk + embed the full context
    chunks = chunk_text(payload.context)
    if chunks:
        chunk_texts, embeddings = build_index(chunks)

        # 3) Retrieve top-k chunks relevant to the last user question
        top_k = 3
        top_chunks = retrieve_top_k(
            question=last_user_question,
            chunk_texts=chunk_texts,
            embeddings=embeddings,
            top_k=top_k,
        )

        # 4) Build a smaller, focused context for Groq
        rag_context = "\n\n".join(top_chunks)
    else:
        # Fallback: if chunking fails, just use the raw context
        rag_context = payload.context

    reply = chat_qa_with_groq(
        context=rag_context,
        messages=[m.model_dump() for m in history],
    )
    return ChatQAResponse(reply=reply)

