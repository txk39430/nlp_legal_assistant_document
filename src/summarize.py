from pathlib import Path
from typing import Literal

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartTokenizerFast,
    BartForConditionalGeneration,
)

# -------------------------
# Paths to fine-tuned models
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINETUNED_T5_DIR = PROJECT_ROOT / "models" / "t5-billsum"
FINETUNED_BART_DIR = PROJECT_ROOT / "models" / "bart-billsum"

# -------------------------
# Load fine-tuned T5
# -------------------------

t5_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_T5_DIR)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_T5_DIR)

# -------------------------
# Load fine-tuned BART
# -------------------------

bart_tokenizer = BartTokenizerFast.from_pretrained(FINETUNED_BART_DIR)
bart_model = BartForConditionalGeneration.from_pretrained(FINETUNED_BART_DIR)


def summarize_with_t5(
    text: str,
    max_new_tokens: int = 400,   
    num_beams: int = 6,          
    min_new_tokens: int = 150,  
) -> str:
    """
    Summarizes using fine-tuned T5 (t5-billsum).
    """
    if not text or not text.strip():
        return ""

    # T5 expects a prefix
    input_text = "summarize: " + text.strip()

    inputs = t5_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,   # keep context capped at 1024 tokens
    )

    output_ids = t5_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,  
        num_beams=num_beams,
        length_penalty=1.0,              # allow natural length
        no_repeat_ngram_size=3,          # avoid repetition like numbers/phrases
        early_stopping=True,
    )

    summary = t5_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return summary.strip()


def summarize_with_bart(
    text: str,
    max_new_tokens: int = 400,   # was 256
    num_beams: int = 6,
    min_new_tokens: int = 150,   # NEW
) -> str:
    """
    Summarize using fine-tuned BART (bart-billsum).
    """
    if not text or not text.strip():
        return ""

    input_text = text.strip()

    inputs = bart_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    output_ids = bart_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,  
        num_beams=num_beams,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    summary = bart_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return summary.strip()


def summarize_text(
    text: str,
    max_new_tokens: int = 400,            # bump default for API
    engine: Literal["t5", "bart"] = "t5",
) -> str:
    """
    Wrappering used by FastAPI.

    engine = "t5"   -> fine-tuned T5 (t5-billsum)
    engine = "bart" -> fine-tuned BART (bart-billsum)
    """
    if engine == "bart":
        return summarize_with_bart(text, max_new_tokens=max_new_tokens)
    # default to t5
    return summarize_with_t5(text, max_new_tokens=max_new_tokens)
