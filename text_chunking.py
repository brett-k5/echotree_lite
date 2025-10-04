# Install required packages first:
# pip install spacy sentence-transformers nltk

import nltk
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('punkt')

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Load a small sentence-transformer model for semantic similarity
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to split text into paragraphs
def split_paragraphs(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs

# Function to compute embeddings for each paragraph
def embed_paragraphs(paragraphs):
    embeddings = embed_model.encode(paragraphs)
    return embeddings

# Function to merge paragraphs based on semantic similarity
def merge_semantic_chunks(paragraphs, embeddings):
    chunks = []
    current_chunk = []
    chunk_embeddings = [] 
    current_embedding = None
    current_word_count = 0

    for i, para in enumerate(paragraphs):
        para_words = len(para.split())
        para_embedding = embeddings[i]

        # If current chunk is empty, start a new one
        if not current_chunk:
            current_chunk = [para]
            current_word_count = para_words
            chunk_embeddings = [para_embedding]
            continue

        # Compute cosine similarity with current chunk embedding
        cos_sim = np.dot(current_embedding, para_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(para_embedding))

        # Merge paragraph if under max_words or semantically similar
        tiers = [
            (550, 0.9),
            (500, 0.75),
            (400, 0.6),
            (300, 0.5)
        ]
        merged = False 
        for max_words, threshold in tiers:
            if current_word_count + para_words <= max_words and cos_sim >= threshold:
                current_chunk.append(para)
                chunk_embeddings.append(para_embedding)
                # Update chunk embedding as mean of embeddings
                current_embedding = np.mean(chunk_embeddings, axis=0)
                current_word_count += para_words
                merged = True
                break
        if not merged:
            # Finish current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_embedding = para_embedding
            current_word_count = para_words

    # Append last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

