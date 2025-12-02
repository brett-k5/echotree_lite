
import { JWT } from 'google-auth-library';
import { supabase } from "./supabaseServer.js";
import type { VercelRequest, VercelResponse } from '@vercel/node';
import fetch, { Response as FetchResponse } from 'node-fetch';

type MatchRow = { text_preview: string, similarity: number, chunk_index: number };
type MatchRowShot = { id: string, shot_example: string, similarity: number}
type MatchRowPast = {id: string, 
                     user_id: string, 
                     echo_id: string, 
                     user_message: string, 
                     echo_response: string, 
                     similarity_user: number, 
                     similarity_echo: number,
                     interaction_index: number
                    }


interface CurrentExchange {
  previousUserMessage: string;
  previousEchoResponse: string;
  previousTextContent: string[];
};

function normalizeVector(vec: number[]): number[] {
      const norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
      return vec.map(val => val / norm);
};

async function embeddingAPICall(token: string, text: string, taskType: string): Promise<FetchResponse> {
  const response = await fetch(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({
          content: { parts: [{ text: text }] },
          outputDimensionality: 768,
          taskType: taskType
        })
      }
    );
  return response
}

async function matchDocs(
      sql_function_name: string, 
      match_count: number,
      userEmbedding: number[],
      user_id: string | null = null,
      echo_id: string | null = null
    ): Promise<MatchRow[] | MatchRowPast[] | null> {
      const params: Record<string, any> = {
        query_embedding: userEmbedding,
        match_count: match_count 
      };

      //Add user and echo id if provided
      if (user_id) params.user_id_input = user_id;
      if (echo_id) params.echo_id_input = echo_id;

      const { data: matchesContent, error} = await supabase.rpc(sql_function_name, params) as {
        data: MatchRow[] | MatchRowPast[] | null;
        error: any;
      };

      if (matchesContent) return matchesContent;
      if (error) console.error("Error fetching matches:", error);

      return null;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try { 
    const rawKey = process.env.GOOGLE_PRIVATE_KEY!;
    const key = rawKey.trim();
    if (!key || !key.startsWith('-----BEGIN PRIVATE KEY-----')) {
      throw new Error("GOOGLE_PRIVATE_KEY is missing or malformed");
    }
    console.log("Newline count:", key?.split('\n').length)

    const auth = new JWT({
      email: process.env.GOOGLE_CLIENT_EMAIL,
      key: key,
      scopes: ['https://www.googleapis.com/auth/generative-language']
    });

    const accessToken = await auth.authorize();
    const token = accessToken.access_token;
    console.log("Access token:", token?.slice(0, 20));

    const { userMessage, currentConversation = [] } = req.body;
    if (!userMessage || typeof userMessage !== "string") {
      return res.status(400).json({error: "Invalid user message"});
    }

    console.log("Current conversation full:", currentConversation)

    let previousUserMessage: string = "";
    let previousEchoResponse: string = "";
    let previousTextContent: string[] = [];
    
    let currentConvContent = "";
    let textContentBlocks = "";
    
    // Change the value based on how much memory of current convo you want the echo to have.
    const exchanges = currentConversation.slice(-2)

    console.log("Current exchange content prior to extraction and formatting:", exchanges)

    if (exchanges.length > 0) {
      currentConvContent = exchanges.map((turn: CurrentExchange, index: number) =>
        `Turn ${index + 1}:\n` +
        `**USER** ${turn.previousUserMessage}\n` +
        `**GEMINI RESPONSE** ${turn.previousEchoResponse}`).join('\n\n--- TURN SEPARATOR ---\n\n');

      textContentBlocks = exchanges.filter(turn => turn.previousTextContent.length > 0)
        .map((turn: CurrentExchange) =>
        `text content used for recent exchanges:\n${turn.previousTextContent.join('\n')}`
        );
    }

    console.log("Current exchange history:", currentConvContent)

    const embeddingResponse = await embeddingAPICall(token, userMessage!, "RETRIEVAL_QUERY")

    const embeddingJson = await embeddingResponse.json();
    if (!embeddingJson.embedding?.values) {
      console.error("Embedding response:", JSON.stringify(embeddingJson, null, 2));
      throw new Error("Gemini embedding response is missing 'values'");
    }
    const userEmbedding = normalizeVector(embeddingJson.embedding.values);

    const fewShotsShort = await matchDocs("match_docs_few_shot_short", 2, userEmbedding) as MatchRowShot[] | null

    const fewShotsLong = await matchDocs("match_docs_few_shot_long", 1, userEmbedding) as MatchRowShot[] | null

    if (req.body.user_id || req.body.echo_id) {
      console.log(`req.body IS properly extracting user_id: ${req.body.user_id} and echo_id: ${req.body.echo_id}`)
    }
    else {
      console.log("req.body is NOT properly extracting user_id and echo_id")
    }

    const userId = req.body.user_id
    const echoId = req.body.echo_id

    console.log(`user_id ${userId}\echo_id: ${echoId}`)




    const matchesPast = await matchDocs("match_past_exchanges", 3, userEmbedding, userId, echoId) as MatchRowPast[] | null

    const matchesContent = await matchDocs("match_documents", 7, userEmbedding) as MatchRow[] | null;

    // Process few shots for RAG prompt
    const fewShotsShortText = (fewShotsShort ?? [])
      .map(f => f.shot_example)
      .join("\n\n--- FEW SHOT SEPARATOR ---\n\n");
    const fewShotsLongText = (fewShotsLong ?? [])
      .map(f => f.shot_example)
      .join("\n\n--- FEW SHOT SEPARATOR ---\n\n");

    const pastExchanges = (matchesPast ?? [])
      .sort((a, b) => a.interaction_index - b.interaction_index)
      .map((f, i, arr) => {
        const prev = arr[i - 1];
        const gap = prev && f. interaction_index !== prev.interaction_index + 1;
        const separator = gap
          ? "\n\n---[CONTEXTUAL GAP: DO NOT INFER CONNECTION]---\n\n"
          : "\n\n";
        return `${separator}\n**USER** ${f.user_message}\n**GEMINI RESPONSE** ${f.echo_response}`;
      }).join("");
    
      console.log("Past interactions on this topic:", pastExchanges)
    
    const matchesArray = [matchesContent, fewShotsShort, fewShotsLong]


    for (const matches of matchesArray) {
      if (Array.isArray(matches)) {
        console.log("Matches found!");
      }
      else if (matches === null) {
        console.log("No matches found.")
      }
    }
    const thresholds = [0.55]
    let filtered: MatchRow[] = []

    for (const t of thresholds) {
      // 1st >= is arrow function, 2nd is greater than or equal to
      filtered = (matchesContent || []).filter((m: MatchRow) => m.similarity >= t);
      if (filtered.length > 0) break; // stop at the highest threshold that returns results)
    }

    let ordered: MatchRow[] = [] 
    if (filtered) {
      ordered = filtered.sort((a, b) => a.chunk_index - b.chunk_index)
    }

    console.log(`Number of chunks returned: ${ordered.length}`)

    let textContent: string[] = []; 
    for (let i = 0; i < ordered.length; i++) {
      const currentChunk = ordered[i]
      console.log("Chunk indices:", currentChunk.chunk_index)
      textContent.push(currentChunk.text_preview);
      
      // Check for a gap between the current chunk and the next one
      if (i < ordered.length - 1) {
        const nextChunk = ordered[i + 1];

        // if the next chunk's index is not consecutive to the current one
        if (nextChunk.chunk_index !== currentChunk.chunk_index + 1) {
          textContent.push("\n\n ---[CONTEXTUAL GAP: DO NOT INFER CONNECTION]---\n\n");
        }
        else {
          textContent.push("\n\n");
        }
      }
    }

    console.log("Chunks returned for current user message", textContent)
  
    if (textContent.length === 0) {
      return res.status(200).json({
        previousUserMessage: userMessage,
        previousEchoResponse: "I have no idea what you are talking about",
        previousTextContent: [],
        echoResponse: "I have no idea what you are talking about"
      });
    }

    // 3. Build prompt
    const prompt = `
You are impersonating the narrator of the events recounted under **BACKGROUND INFO** and **TEXT CONTENT USED**. Adopt the perspective, style, and tone as established by the examples under **RESPONSE STYLE**.

**Conversational Context**: Use **RECENT EXCHANGES** and **PAST INTERACTIONS** to maintain conversational context and **avoid repetition** of **factual information** and **phrasing** already stated in **GEMINI RESPONSE** under **RECENT EXCHANGES** and **GEMINI RESPONSE** under **PAST INTERACTIONS** in your response.

**DO NOT** use the phrase "alright, alright, alright" in your response unless it appears in the responses under **RESPONSE STYLE**

You may have had previous interactions with the user on the topic you are currently discussing. If you did, those interactions will appear under **PAST INTERACTIONS**. If the interactions documented under **PAST INTERACTIONS** are relevant to the topic you are discussing with the user, take them into account in your response to the user. Otherwise, ignore them.

The information under **BACKGROUND INFO** and **TEXT CONTENT USED** consists of events and facts that took place in the **past** (multiple years ago).
-**DO NOT** discuss these past events as though they are happening now or within the last few days.
-You are currently in a conversation with the user. Ensure your responses are **conversational, natural, and grammatically correct** in the present tense, reflecting on past events where relevant.

-**ALWAYS** prioritize the conversational flow and the established **style and tone**.
=IF the user's message is an explicit request to continue or expand on the immediate previous topic (e.g., "tell me more," "what happened next," "expand on that"), YOUR RESPONSE MUST return to the last narrative topic.
-**DO NOT** provide any more information than is needed to answer the user's message in the conversational style of the examples under **RESPONSE STYLE**
-**DO NOT** mention or refer to the sections **BACKGROUND INFO** or **TEXT CONTENT USED** in your response.
-**IF** the **BACKGROUND INFO** is empty, you must assume there is no source material for the narrator's past life. When answering, strictly adhere to the **No Direct Answer** guidelines, especially for factual questions about the narrator.

### Information Usage Guidelines:
1.**Direct Relevance:** Use facts from **BACKGROUND INFO** and **TEXT CONTENT USED** to directly answer the user's question only if they are relevant.
2.**No Direct Answer:** If the RAG content (**BACKGROUND INFO** / **TEXT CONTENT USED**) does not contain the answer:
   * **Factual:** Answer based on **accessible, non-specialized general knowledge** (e.g., basic math, basic science, geography, widely known current events). **DO NOT** answer questions that require deep or specialized domain knowledge (e.g., advanced physics, complex historical analysis, niche statistics). **DO NOT** use common knowledge or external facts specific to the narrator's life (e.g., their public biography, filmography, or known opinions outside the provided text). If you cannot answer with confidence based on RAG content **(which includes empty BACKGROUND INFO)** or accessible general knowledge, state you are **not sure**. **DO NOT** make up facts.
   * **Opinion/Inference:** Infer the narrator's opinion based *only* on the provided content and the **RESPONSE STYLE**. Use this inferred opinion to answer.

---
**BACKGROUND INFO:**
"""
${textContent.join("\n\n")}
"""
**RECENT EXCHANGES:**
"""
${currentConvContent}
"""
**TEXT CONTENT USED**
"""
${textContentBlocks}
"""
**PAST INTERACTIONS**
"""
${pastExchanges}
**RESPONSE STYLE:** (Few-Shot Examples)
"""
${fewShotsShortText}
${fewShotsLongText}
"""
**USER:**
"""
${userMessage}
"""
**YOUR RESPONSE:**
`;

      

    // call Gemini model with context
    const response = await fetch(
      "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
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
    const echoResponse = resultJson.candidates?.[0]?.content?.parts?.[0]?.text;
    
    previousUserMessage = userMessage
    previousEchoResponse = echoResponse 
    previousTextContent = textContent

    const prevUserMessage = await embeddingAPICall(token, previousUserMessage!, "RETRIEVAL_DOCUMENT")
    const prevEchoResponse = await embeddingAPICall(token, previousEchoResponse!, "RETRIEVAL_DOCUMENT")

    const prevUserMessageJson = await prevUserMessage.json()
    const prevEchoResponseJson = await prevEchoResponse.json()

    if (!prevUserMessageJson.embedding?.values) {
      console.error("Previous user message embedding response:", JSON.stringify(prevUserMessageJson, null, 2));
      throw new Error("Gemini embedding response is missing 'values'");
    }

    if (!prevEchoResponseJson.embedding?.values) {
      console.error("Previous echo response embedding response:", JSON.stringify(prevEchoResponseJson, null, 2));
      throw new Error("Gemini embedding response is missing 'values'");
    }
    const prevMessageEmbedding = normalizeVector(prevUserMessageJson.embedding.values);
    const prevResponseEmbedding = normalizeVector(prevEchoResponseJson.embedding.values);

    
    await supabase.from('echo_responses').insert([{
      user_mess_vec: prevMessageEmbedding,
      echo_resp_vec: prevResponseEmbedding
    }])
    
    return res.status(200).json({
      previousUserMessage,
      previousEchoResponse,
      previousTextContent,
      echoResponse
    });
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

