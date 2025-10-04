# Standard imports 
import os
import json

# Third party imports
import numpy as np
import pandas as pd

# Local Application imports 
from python_scripts.chunking_utilities import split_paragraphs, embed_chunks, merge_semantic_chunks

# Example usage
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", "greenlights.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_chunks = split_paragraphs(text)
    embeddings = embed_chunks(text_chunks)
    semantic_chunks = merge_semantic_chunks(text_chunks, embeddings)

    print(f"Created {len(semantic_chunks)} semantic-aware chunks.")
    for i in range(10):
        print(f"Sample chunk {i}:\n", semantic_chunks[i])
    
    chunk_lengths = []
    for chunk in semantic_chunks:
        chunk_length = len(chunk.split())
        chunk_lengths.append(chunk_length)

    chunk_lengths = np.array(chunk_lengths)
    print(f"Smallest chunk: {chunk_lengths.min()} words")
    print(f"Longest chunk: {chunk_lengths.max()} words")

    pd.set_option("display.max_rows", None)

    chunk_lengths = pd.Series(chunk_lengths)
    print(f"Chunk lengths: {chunk_lengths.value_counts()}")

    out_path = os.path.join(os.path.dirname(__file__), "greenlights_chunks.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(semantic_chunks, f, ensure_ascii=False, indent=2)
