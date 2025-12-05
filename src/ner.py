#spaCy uses a highly efficient, statistical Named Entity Recognition (NER) model based on a deep convolutional neural network and a transition-based approach to identify and label real-world objects in text

import spacy
from functools import lru_cache

MODEL_NAME = "en_core_web_sm"


@lru_cache(maxsize=1)
def load_ner_model():
    """
    Load spaCy NER model once and cache it.
    """
    print(f"Loading spaCy NER model: {MODEL_NAME}")
    return spacy.load(MODEL_NAME)


def extract_entities(text: str):
    """
    Extract named entities from text using spaCy NER.
    Returns a list of entity dictionaries.
    """
    nlp = load_ner_model()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })

    return {"entities": entities}
