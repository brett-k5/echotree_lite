# Standard Library Imports
import json
import os

# Third party imports
from dotenv import load_dotenv
from google import genai 
from google.genai import types
import numpy as np
from numpy.linalg import norm
from supabase import create_client, Client


chunks_path = os.path.join(os.path.dirname(__file__), "greenlights_chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    greenlights_chunks = json.load(f)

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

batch_1 = greenlights_chunks[0:97]
batch_2 = greenlights_chunks[97:194]
batch_3 = greenlights_chunks[194:291]
batch_4 = greenlights_chunks[291:388]
batch_5 = greenlights_chunks[388:485]
batch_6 = greenlights_chunks[485:582]

batches = [batch_1, batch_2, batch_3, batch_4, batch_5, batch_6]

embeddings = []
for batch in batches:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch,
        config=types.EmbedContentConfig(output_dimensionality=768, task_type="RETRIEVAL_DOCUMENT")
    )
    for embedding in response.embeddings:
        embedding_values_np = np.array(embedding.values)
        normed_embedding = embedding_values_np /np.linalg.norm(embedding_values_np)
        json_compatible_embedding = normed_embedding.tolist()
        embeddings.append(json_compatible_embedding)


print(f"Generated embeddings for {len(embeddings)} chunks.")

out_path = os.path.join(os.path.dirname(__file__), "greenlights_embeddings.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=2)

print(f"Saved {len(embeddings)} embeddings to {out_path}")

