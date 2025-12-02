# Standard Library Imports
import json
import os

# Third party imports
from dotenv import load_dotenv
from supabase import create_client, Client

chunks_path = os.path.join(os.path.dirname(__file__), "few_shot_long_chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    few_shot_chunks = json.load(f)

embeddings_path = os.path.join(os.path.dirname(__file__), "few_shot_long_embeddings.json")
with open(embeddings_path, "r", encoding="utf-8") as f:
    embeddings = json.load(f)

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

echo_name = "Matthew" # set echo_name for all embeddings
data = []
for i, (chunk, vec) in enumerate(zip(few_shot_chunks, embeddings)):
    print(f"chunk type: {type(chunk)}, vec type: {type(vec)}")
    row = {
        "embedding": vec,
        "chunk_index": i + 1,                   
        "echo_name": echo_name,
        "shot_example": chunk
    }
    data.append(row)

res = supabase.table("greenlights-few-shots-long").insert(data).execute()
