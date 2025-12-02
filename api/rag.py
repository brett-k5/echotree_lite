# /api/rag.py
from http.server import BaseHTTPRequestHandler
import json
import os
import traceback

# Third party installations
from google import genai
import numpy as np
from supabase import create_client, Client

configure(api_key=os.environ["GEMINI_API_KEY"])

chat_model = GenerativeModel("gemini-1.5-flash")

# Initialize Gemini client
embed_model = genai.embed_content
configure(api_key=os.environ["GEMINI_API_KEY"])
chat_model = GenerativeModel("gemini-2.0-flash")

# Supabase client setup
supabase_url = os.environ["SUPABASE_URL"]
supabase_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
client = supabase.create_client(supabase_url, supabase_key)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Validate Content-Type
            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                self.send_response(415)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Unsupported Media Type",
                    "expected": "application/json"
                }).encode())
                return

            # Read and decode request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Try parsing JSON
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Invalid JSON",
                    "type": "JSONDecodeError"
                }).encode())
                return

            user_message = data.get("userMessage", "")

            # Embed user message
            embedding_response = embed_model(
                model="models/embedding-001",
                content=user_message,
                task_type="retrieval_document",
                output_dimensionality=256
            )
            query_vector = embedding_response["embedding"]["values"]
            print("Embedding response:", embedding_response)  # ✅ replaced logging.info

            # Query Supabase for top-k matches
            match_response = client.rpc("match_documents", {
                "query_embedding": query_vector,
                "match_count": 10
            }).execute()
            print("Supabase match response:", match_response.data)  # ✅ replaced logging.info

            # Format retrieved context
            context_blocks = [
                f"- {doc['content']}" for doc in match_response.data
            ]
            context = "\n".join(context_blocks)

            # Construct prompt
            prompt = f"""
            Use the context provided in "CONTEXT:" to answer the user's message provided in "USER:".

            - Answer as though you are the narrator of the events in "CONTEXT:".
            - Adopt the style and tone of the narrator events in "CONTEXT:" as though it were your own in your answer.
            - Keep your answers short unless you think the message under "USER:" really calls for a longer response.
            - If the message under "USER:" calls for a factual answer:
                - answer it based on information you would expect most people to know, and on the information in "CONTEXT:" if the information in "CONTEXT:" seems relevant.
                - If the answer is not something most people would know:
                    - respond using the information from context if the context fully answers the question,
                    - If the context is partially relevant, do your best to answer based on context and information most people would know, but indicate uncertainty where needed.
                    - If the context does NOT answer the message adequately, say that you are not sure, and steer the conversation in a different direction.
            - DO NOT EVER indicate that you are basing your answers off of the info provided in "CONTEXT:" in your answer.
            - Make your response conversational, natural, and always in the style and tone of the narrator of the events under "CONTEXT:"

            CONTEXT:
            \"\"\"
            {context}
            \"\"\"

            USER:
            \"\"\"
            {user_message}
            \"\"\"
            """

            # Generate response
            gemini_response = chat_model.generate_content(prompt)
            reply = gemini_response.text

            # ✅ Return hardcoded response to confirm invocation
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "text": reply,
                "echo": data
            }).encode())

        except Exception:
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Internal server error",
                "type": "Exception"
            }).encode())
