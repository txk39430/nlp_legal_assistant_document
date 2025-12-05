from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from .config import QA_MODEL_NAME, QA_MAX_CONTEXT_LENGTH


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("QA: Using Apple MPS device")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("QA: Using CUDA GPU")
        return torch.device("cuda")
    print("QA: Using CPU")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_qa_model_and_tokenizer():
    """
    Loading QA model + tokenizer once and cache them.
    """
    print(f"Loading QA model: {QA_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)

    device = _get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device

def answer_question(question: str, context: str) -> dict:
    """
    Given a question and a context (legal/policy text), return the best answer span.

    Returns:
        {
            "answer": str,
            "score": float,
            "start": int,
            "end": int
        }
    """
    tokenizer, model, device = load_qa_model_and_tokenizer()

    encoded = tokenizer(
        question,
        context,
        truncation=True,
        max_length=QA_MAX_CONTEXT_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    # Converting logits to probabilities
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    # start/end index
    start_idx = int(torch.argmax(start_probs, dim=-1)[0])
    end_idx = int(torch.argmax(end_probs, dim=-1)[0])

    if end_idx < start_idx:
        end_idx = start_idx

    input_ids = encoded["input_ids"][0]
    answer_ids = input_ids[start_idx : end_idx + 1]

    # Decoding while skipping special tokens like <s>, </s>, <pad>, etc.
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # Confidence score: average of start/end probs at chosen indices
    start_score = float(start_probs[0, start_idx].item())
    end_score = float(end_probs[0, end_idx].item())
    score = (start_score + end_score) / 2.0

    # Heuristic: if answer is empty or looks like junk, treat as "no answer"
    if not answer or answer in {"<s>", "</s>", tokenizer.cls_token, tokenizer.sep_token}:
        answer = ""
    # Optional: threshold – if score too low, say no answer
    NO_ANSWER_THRESHOLD = 0.25
    if score < NO_ANSWER_THRESHOLD:
        answer = ""

    return {
        "answer": answer,
        "score": score,
        "start": start_idx,
        "end": end_idx,
    }
