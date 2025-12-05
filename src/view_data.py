from datasets import load_dataset
import pandas as pd
import nltk

# This Make sure punkt tokenizer is available
nltk.download("punkt")

# ---- Load ENTIRE BillSum dataset ----
# train: ~18,949 docs
# test: ~3,269 docs
# ca_test: ~1,237 docs
ds = load_dataset("billsum", split="train")

# Convert to pandas
df = pd.DataFrame(ds)

print("\n============================")
print(" FULL DATASET SHAPE ")
print("============================")
print(df.shape)        # (num_documents, 3)
print(df.columns)      # ['text', 'summary', 'title']

# ----------------------------
# COUNT SENTENCES IN EACH DOCUMENT
# ----------------------------
def count_sentences(text):
    return len(nltk.sent_tokenize(text))

df["sentence_count"] = df["text"].apply(count_sentences)

print("\n============================")
print(" TOTAL SENTENCE COUNT ")
print("============================")
print(df["sentence_count"].sum())

print("\n============================")
print(" HEAD (first 5 rows) ")
print("============================")
print(df.head())

# Save a CSV preview (optional)
df.head(20).to_csv("bill_sum_preview.csv", index=False)
print("\nSaved: bill_sum_preview.csv")
