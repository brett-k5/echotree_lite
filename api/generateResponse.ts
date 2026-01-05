
import { JWT } from 'google-auth-library';
import { supabase } from "./supabaseServer.js";
import type { VercelRequest, VercelResponse } from '@vercel/node';
import fetch, { Response as FetchResponse } from 'node-fetch';

// Define types for returned content from supabase following vector similarity search
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

// Define CurrentExchange type
interface CurrentExchange {
  previousUserMessage: string;
  previousEchoResponse: string;
  previousTextContent: string[];
};

// Define a function to normalize vector embeddings
function normalizeVector(vec: number[]): number[] {
      const norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
      return vec.map(val => val / norm);
};

// Define function for Gemini embedding API call
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

// Define function to trigger SQL based vector similarity search
async function matchDocs(
      sql_function_name: string, 
      match_count: number,
      userEmbedding: number[],
      user_id: string | null = null,
      echo_id: string | null = null
    ): Promise<MatchRow[] | MatchRowPast[] | null> {

      // Define parameters for SQL function to be applied in Supabase
      const params: Record<string, any> = {
        query_embedding: userEmbedding,
        match_count: match_count 
      };

      //Add user and echo id if provided
      if (user_id) params.user_id_input = user_id;
      if (echo_id) params.echo_id_input = echo_id;

      // Call SQL supabase embedding similarity search function and assign result to matchesContent variable
      const { data: matchesContent, error} = await supabase.rpc(sql_function_name, params) as {
        data: MatchRow[] | MatchRowPast[] | null;
        error: any;
      };
      
      // Return retrieved content or log an error if no embeddings are returned
      if (matchesContent) return matchesContent;
      if (error) console.error("Error fetching matches:", error);

      return null;
}

// Make this function available to other modules (but specifically chat.ts)
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
    
    // Desctructure the response from chat.ts
    const { userMessage, currentConversation = [] } = req.body; // unpack content of the request from chat.ts into userMessage and currentConversation variables. If no prior conversation contnent, currentConversation will default to an empty array
    if (!userMessage || typeof userMessage !== "string") {     // handle an invalid user message
      return res.status(400).json({error: "Invalid user message"});
    } 

    console.log("Current conversation full:", currentConversation)
    
    // Initialize placeholder variables (these variables all exist in the currentConversation portino of the Vercel request, but
    // We need to intialize them with type definitions before accessing them there for the sake of type safety)
    let previousUserMessage: string = "";
    let previousEchoResponse: string = "";
    let previousTextContent: string[] = [];
    
    let currentConvContent = "";
    let textContentBlocks = "";
    
    
    const exchanges = currentConversation.slice(-2) // Change the value based on how much memory of current convo you want the echo to have.

    console.log("Current exchange content prior to extraction and formatting:", exchanges)

    // Format exchange history of current context window
    if (exchanges.length > 0) {
      currentConvContent = exchanges.map((turn: CurrentExchange, index: number) => //iterate over exchanges variable
        `Turn ${index + 1}:\n` + // make order of exchanges clear for the prompt
        `**USER** ${turn.previousUserMessage}\n` +
        `**GEMINI RESPONSE** ${turn.previousEchoResponse}`).join('\n\n--- TURN SEPARATOR ---\n\n'); // make clear where turns end
                                                                                                    // and where they begin
      // include text basis for previous exchanges in prompt (for each exchange)
      textContentBlocks = exchanges.filter(turn => turn.previousTextContent.length > 0) // include only tokens where text content is present
        .map((turn: CurrentExchange) =>
        `text content used for recent exchanges:\n${turn.previousTextContent.join('\n')}`  // label text content to differentiate from previous exchanges 
        );
    }
    
    // Log conversation from current context window in Vercel
    console.log("Current exchange history:", currentConvContent)
    
    // Embed user message as a retrieval query
    const embeddingResponse = await embeddingAPICall(token, userMessage!, "RETRIEVAL_QUERY")
    
    // Parse json string and assign to embeddingJson variable
    const embeddingJson = await embeddingResponse.json();
    if (!embeddingJson.embedding?.values) { 
      console.error("Embedding response:", JSON.stringify(embeddingJson, null, 2)); // log error if there are no values in the embedding object
      throw new Error("Gemini embedding response is missing 'values'");
    }
    // Normalize the vector embeddings
    const userEmbedding = normalizeVector(embeddingJson.embedding.values);
    
    // Call matchDocs function on user embedding to retrieve short few shot examples
    const fewShotsShort = await matchDocs("match_docs_few_shot_short", 2, userEmbedding) as MatchRowShot[] | null
    
    // Call matchdocs function on user embedding to retrieve long few shot examples
    const fewShotsLong = await matchDocs("match_docs_few_shot_long", 1, userEmbedding) as MatchRowShot[] | null

    // Check to ake sure user_id and echo_id variables are being extracted from the Vercel request properly
    if (req.body.user_id || req.body.echo_id) {
      console.log(`req.body IS properly extracting user_id: ${req.body.user_id} and echo_id: ${req.body.echo_id}`)
    }
    else {
      console.log("req.body is NOT properly extracting user_id and echo_id")
    }
    
    // Assign variables from Vercel request
    const userId = req.body.user_id
    const echoId = req.body.echo_id

    console.log(`user_id ${userId}\echo_id: ${echoId}`)



    // Assign past discussions relevant to current topic to matchesPast
    const matchesPast = await matchDocs("match_past_exchanges", 3, userEmbedding, userId, echoId) as MatchRowPast[] | null
    
    // Assign Greenlights which was returned in response to the Retrieval query (which was the user message) to matchesContent variable
    const matchesContent = await matchDocs("match_documents", 7, userEmbedding) as MatchRow[] | null;

    // Process few shots for RAG prompt
    const fewShotsShortText = (fewShotsShort ?? [])
      .map(f => f.shot_example)
      .join("\n\n--- FEW SHOT SEPARATOR ---\n\n");
    const fewShotsLongText = (fewShotsLong ?? [])
      .map(f => f.shot_example)
      .join("\n\n--- FEW SHOT SEPARATOR ---\n\n");

      // Make sure past exchanges appear in correct order
      const pastExchanges = (matchesPast ?? [])
      .sort((a, b) => a.interaction_index - b.interaction_index)
      .map((f, i, arr) => {
        // If previous exchanges are not continuous, note in the prompt
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
    const thresholds = [0.55] // Set vector similarity standard for retrieved Greenlights text
    let filtered: MatchRow[] = []

    // Filter for Greenlights text matches that exceed threshold.
    for (const t of thresholds) {
      // 1st >= is arrow function, 2nd is greater than or equal to
      filtered = (matchesContent || []).filter((m: MatchRow) => m.similarity >= t);
      if (filtered.length > 0) break; // stop at the highest threshold that returns results)
    }

    let ordered: MatchRow[] = [] 
    if (filtered) {
      // sort Greenlights text by chunk index (we want everything to appear in the order it appears in the book)
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
        previousEchoResponse: "I have no idea what you are talking about", // MaccOnaughey has no experience recounted in Greenlights that 
        // is relevant to what the user is asking
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
              parts: [{text: prompt}] // pass prompt to Gemini's text Generator 
            }
          ]
        })
      }
    )

    const resultJson = await response.json();
    console.log("Gemini raw response:", JSON.stringify(resultJson, null, 2));
    const echoResponse = resultJson.candidates?.[0]?.content?.parts?.[0]?.text;
    
    // Assign userMessage, echoResponse, and textContent variables from this round of responses 
    // to be embedded with Gemini embeddings. Ultimately will be returned as part of Vercel response
    // with new variable names (see below). This is how we maintain the context window - each response is 
    // sent back to chat.ts added to currentConv
    previousUserMessage = userMessage
    previousEchoResponse = echoResponse 
    previousTextContent = textContent
    
    // Embed user message (now previousUserMessage) and echo response (now previousEchoResponse)
    const prevUserMessage = await embeddingAPICall(token, previousUserMessage!, "RETRIEVAL_DOCUMENT")
    const prevEchoResponse = await embeddingAPICall(token, previousEchoResponse!, "RETRIEVAL_DOCUMENT")

    // Parse json strings
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
    
    // Nornalize vector embeddings
    const prevMessageEmbedding = normalizeVector(prevUserMessageJson.embedding.values);
    const prevResponseEmbedding = normalizeVector(prevEchoResponseJson.embedding.values);

    // Insert user message and echo response from this round to be used for context in future exchanges
    await supabase.from('echo_responses').insert([{
      user_mess_vec: prevMessageEmbedding,
      echo_resp_vec: prevResponseEmbedding
    }])
    
    // Send user message, echo response, and text content back to chat.ts to build context window
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

