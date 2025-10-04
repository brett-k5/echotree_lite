import { GoogleGenerativeAI } from "@google/generative-ai";
import { supabase } from "./supabaseServer.js";
import type { VercelRequest, VercelResponse } from '@vercel/node';
import fetch from 'node-fetch';


export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    const geminiKey = process.env.GEMINI_API_KEY;
    if (!geminiKey) {
      throw new Error("GEMINI_API_KEY is not defined in environment variables");
    }
    const genAI = new GoogleGenerativeAI(geminiKey);

    const { userMessage } = req.body;

    if (!userMessage || typeof userMessage !== "string") {
      return res.status(400).json({error: "Invalid user message"});
    }

    // 1. Embed user message
    const model = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
    const result = await model.embedContent(userMessage);
    const userEmbedding = result.embedding.values;

    // 2. Query Supabase for matching chunks
    const { data: matches, error } = await supabase.rpc("match_documents", {
      query_embedding: userEmbedding,
      match_count: 10
    });
    if (error) {
    console.error("Error fetching matches:", error);

    return res.status(500).json({
      error: "Well, you see, I'm gonna be honest - I don't quite have a map for that one right now. Everything's a little foggy upstairs at the moment"
    });
    }
    const thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.5, -1.0, -1.1]
    let filtered: any[] = []

    for (const t of thresholds) {
      // 1st >= is arrow function, 2nd is greater than or equal to
      filtered = (matches || []).filter((m: any) => m.similarity >= t);
      if (filtered.length > 0) break; // stop at the highest threshold that returns results)
    }

    const contexts = (filtered || []).map((m: any) => m.text_chunk);

    if (contexts.length === 0) {
      return res.status(200).json({
          text: "I have no idea what you are talking about"
      });
    }

    // 3. Build prompt
    const prompt = `
      Use the context provided in "CONTEXT:" to answer the user's message provided in "USER:".
  
      - If the context fully answers the user’s question, respond using that information.
      - If the context is partially relevant, do your best to answer based on what you know, but indicate uncertainty where needed.
      - If the context does NOT answer the question adequately, explain politely that you are not sure and that the answer is unknown.
      - Make your response conversational and natural.
      - Match the style and tone of the context.

      CONTEXT: 
      """
      ${contexts.join("\n\n")}
      """
      USER:
      """
      ${userMessage}
      """
      `;

    // call Gemini model with context
    const response = await fetch(
      "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${geminiKey}`
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [{text: prompt}]
            }
          ]
        })
      }
    )


    const resultJson = await response.json();
    console.log("Gemini raw response:", JSON.stringify(resultJson, null, 2));
    const generatedText = resultJson.candidates?.[0]?.content?.parts?.[0]?.text;
    
    return res.status(200).json({ text: generatedText });
  } 
  catch (err) {
    const error = err as Error 
    console.error("generateResponse failed:", error.message);
    return res.status(500).json({
      error: "Internal Server Error",
      detail: error.message
    });
  }
}

