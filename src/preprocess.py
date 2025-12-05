from typing import Optional
from datasets import load_dataset
from .config import BILLSUM_SPLIT


def load_billsum(split: str = BILLSUM_SPLIT,
                 subset_slice: Optional[str] = None):
    """
    Load the BillSum dataset from Hugging Face.

    Args:
        split: which split to load (e.g., "ca_test" or "train").
        subset_slice: optional slicing like "0:1000" to load a subset
                      useful when debugging on a small sample.

    Returns:
        A HuggingFace Dataset object with fields: "text", "summary", "title".
    """
    hf_split = split
    if subset_slice is not None:
        # Example: split="ca_test", subset_slice="0:2000" -> "ca_test[0:2000]"
        hf_split = f"{split}[{subset_slice}]"

    print(f"Loading BillSum dataset with split='{hf_split}' ...")
    ds = load_dataset("billsum", split=hf_split)
    print(f"Loaded {len(ds)} examples.")
    return ds


def clean_text(text: str) -> str:
    # Removing leading/trailing whitespace and normalize spaces
    return " ".join(text.split())


if __name__ == "__main__":
    dataset = load_billsum(split=BILLSUM_SPLIT, subset_slice="0:5")

    first = dataset[0]
    raw_text = first["text"]
    raw_summary = first["summary"]

    print("\n--- RAW TEXT (first 500 chars) ---")
    print(raw_text[:500])
    print("\n--- RAW SUMMARY (first 300 chars) ---")
    print(raw_summary[:300])

    print("\n--- CLEANED TEXT (first 300 chars) ---")
    print(clean_text(raw_text)[:300])
