from utils.normalize import normalize_text

def tokenize_text(text):
    normalized_text = normalize_text(text)
    tokens = normalized_text.split()  # Simple space-based tokenization
    return tokens

