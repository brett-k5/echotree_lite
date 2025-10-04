# Standard Library Imports
import json
import os

# Third party imports
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client


chunks_path = os.path.join(os.path.dirname(__file__), "greenlights_chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    greenlights_chunks = json.load(f)

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

embeddings = []
for chunk in greenlights_chunks:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )

    embeddings.append(response.embeddings[0].values)

print(f"Generated embeddings for {len(embeddings)} chunks.")

out_path = os.path.join(os.path.dirname(__file__), "greenlights_embeddings.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=2)

print(f"Saved {len(embeddings)} embeddings to {out_path}")


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

echo_name = "Matthew" # set echo_name for all embeddings
data = []
for i, (chunk, vec) in enumerate(zip(greenlights_chunks, embeddings)):
    print(f"chunk type: {type(chunk)}, vec type: {type(vec)}")
    row = {
        "embedding": vec,
        "chunk_index": i + 1,                   # full chunk
        "echo_name": echo_name,
        "text_preview": chunk
    }
    data.append(row)

res = supabase.table("greenlights-embeddings").insert(data).execute()

if res.error:
    print("Error inserting embeddings:", res.error)
else:
    print(f"Inserted {len(embeddings)} embeddings into supabase")

