# Standard library imports
import re

# Third party imports
import numpy as np
from sentence_transformers import SentenceTransformer


# Load a small sentence-transformer model for semantic similarity
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to split text into paragraphs
def split_paragraphs(text):
    # Split on end-of-sentence punctuation followed by newline(s) OR two or more newlines
    text_chunks = re.split(r'(?<=[.!?])\n+|\n{2,}', text)
    # Strip whitespace and remove empty chunks
    text_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
    return text_chunks

# Function to compute embeddings for each text_chunk
def embed_chunks(text_chunks):
    embeddings = embed_model.encode(text_chunks)
    return embeddings

def merge_semantic_chunks(text_chunks, embeddings, min_words=20):
    merged_chunks = []
    current_chunk = []
    chunk_embeddings = []
    current_embedding = None
    current_word_count = 0

    for i, para in enumerate(text_chunks):
        para_words = len(para.split())
        para_embedding = embeddings[i]

        # If current chunk is empty, start new
        if not current_chunk:
            current_chunk = [para]
            current_word_count = para_words
            chunk_embeddings = [para_embedding]
            current_embedding = para_embedding
            continue

        # Compute cosine similarity
        cos_sim = np.dot(current_embedding, para_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(para_embedding)
        )

        # Tiered cosine similarity thresholds
        tiers = [
            (350, 0.85),
            (300, 0.75),
            (250, 0.6),
            (200, 0.5),
            (150, 0.4),
            (100, 0.2),
            (40, -1.1)
        ]

        merged = False
        for max_words, threshold in tiers:
            if current_word_count + para_words <= max_words and cos_sim >= threshold:
                current_chunk.append(para)
                chunk_embeddings.append(para_embedding)
                current_embedding = np.mean(chunk_embeddings, axis=0)
                current_word_count += para_words
                merged = True
                break

        # Merge tiny chunks by word count
        if not merged:
            if current_word_count < min_words or para_words < min_words:
                current_chunk.append(para)
                chunk_embeddings.append(para_embedding)
                current_word_count += para_words
                current_embedding = np.mean(chunk_embeddings, axis=0)
                merged = True
                continue

        # If not merged, finish current chunk and start new
        if not merged:
            merged_chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            chunk_embeddings = [para_embedding]
            current_embedding = para_embedding
            current_word_count = para_words

    # Append last chunk
    if current_chunk:
        merged_chunks.append(" ".join(current_chunk))

    return merged_chunks