npm install @google/generative-ai

import { GoogleGenerativeAI } from "@google/generative-ai";

// Initialize Gemini client
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);

// Use the embeddings model
const model = genAI.getGenerativeModel({ model: "embedding-001" });

async function embedText(text: string) {
  const result = await model.embedContent({ content: text });
  const embedding = result.embedding.values; // array of floats
  console.log("Embedding length:", embedding.length);
  console.log("First 5 dims:", embedding.slice(0, 5));
  return embedding;
}

// Example usage
embedText("Alright, alright, alright. Just keep livin’.");