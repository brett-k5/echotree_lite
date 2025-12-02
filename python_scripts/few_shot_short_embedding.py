# Standard imports 
import os
import json

# Third party imports
from dotenv import load_dotenv
from google import genai 
from google.genai import types
import numpy as np
from numpy.linalg import norm
import pandas as pd
from supabase import create_client, Client

# Local Application imports 
from python_scripts.chunking_utilities import split_few_shots
 

# Example usage
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", "few-shot-short.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        few_shot_short = f.read()

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    text_chunks = split_few_shots(few_shot_short)

    embeddings = []
    for chunk in text_chunks:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )

        embedding_values_np = np.array(response.embeddings[0].values)
        normed_embedding = embedding_values_np /np.linalg.norm(embedding_values_np)
        json_compatible_embedding = normed_embedding.tolist()
        embeddings.append(json_compatible_embedding)


    print(f"Generated embeddings for {len(embeddings)} chunks.")

    out_path = os.path.join(os.path.dirname(__file__), "few_shot_short_embeddings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(embeddings)} embeddings to {out_path}")

    out_path = os.path.join(os.path.dirname(__file__), "few_shot_short_chunks.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(text_chunks, f, ensure_ascii=False, indent=2)

