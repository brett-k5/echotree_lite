# 🧑‍💻 **Matthew McConaughey Echo**  
This project was intended to be a demo for a company called echotree that is attempting to encode an indvidual's essence with vector embeddings 80,000 words of interview text and Retrieval Augmented Generation Facilitated Interaction. 

🚀 **Project Overview**  
This is a Retrieval Agumented Generation (RAG) pipeline that attempts to mimic the style and personality of Matthew McConaughey.  
I chose Matthew McConaughey as my demo echo because his style and manner of speaking are widely recognizable, making the quality of my RAG pipelie easy to gauge.

This project utilizes TypeScript for front end User Interfacce (UI) functionality, as well as backend Gemini API calls, caching, and dynamic Gemini prompt engineering. I utilized python to chunk and embed the original Greenlights text as well as the few shot examples that are injected into the prompt for styling. 

The text embeddings are saved in supabase and are retrieved via SQL functions that are defined in supabase and called in TypeScrpt functions when a user inputs a message.

📊 **Matthew McCon-Echo's Current Capabilities**   
At present Matthew McCon-Echo (as I have taken to calling him) has a context window that goes back two exchanges into the past (includig the curent exchange therefore, he can consider three exchanges at once). However, he can also remember past exchanges, and every now and again he will reference a conversation you had with him in the past.  

His near term context window can easily be expanded by editing line 123 in the generateResponse.ts script to slice n desired excahanges back instead of 2. However, if we expand the context window dramatically the token count for the prompt will explode and our API costs along with it. 2 seems to be a nice tradeoff because McCon-Echo often answers questions naturally even when he has no context. However, it would be reasonable to consider retaining as many as 5 past eachanges in the context window.

McConaughey's syle in his answers is quite good - even more authentic than if you told a state of the art language model to imitate Matthew McConaughey. However, he occasionally loses the thread of the conversation (this could easily be avoided by expanding his context window, but would come with increasing API costs), and sometimes gives unorthodox, AI giveaway type responses due to his limited 80,000 words worth of life experience to draw on in his answers. He will also occasionally mis-represent events in his life because although the order of the text Gemini is presented in the prompt is always maintained, he does not maintain working knowledge of all 80,000 words across exchanes. Therefore, he may say things in one response that contradict something he said in a previous response because he only has access to the text that was retrieved from the vector similarity search of the current user response and the exchanges immediately preceding the present exchange. Of course, he also cannot give answers about events in the real Matthew McConaughey's life that do not take place in Greenlights, but this is by design. Nevertheless, all things considered, he gives mostly accurate answers and maintains a shocking degree of the spirit that makes Matthew McConaughey's style of communication so recognizable.

📊 **A Brief Overview of how Matthew McCon-Echo works**   
To chunk the text from Greenlights appropriately, each paragraph from the text was embedded via sentence_transformer's SentenceTransformer() method. Note, that these embeddings were only utilized for chunking purposes and played no role in the retrieval searches that serve as the basis for McCon-Echo's output. Therefore, it is not necessary to utilize Gemini embeddings here, and it would only add unnecessary costs to do so. Chunks were then combined or kept apart based on a sliding scale of vector simiarlity. The lower the number of words for a pair of chunks, the lower the similarity threshold was for merging them. See the merge_semantic_chunks function in chunking_utilities.py for more details. Once the text had been chunked it was embedded again - this time with Gemini's embedding API call - we created 768 dimensional embeddings of type "RETRIEVAL DOCUMENT". Briefly, it is worth noting that when ever embeddings are being saved to supabse they are embedded as "RETRIEVAL DOCUMENTS" and whenever they are embedded to be utilized as the basis for similarity search they are embedded as a "RETRIEVAL QUERY". 

The meat and potatoes of the RAG system is implemented via the generateResponse.ts script and the chat.ts script. generateResponse.ts exports the handler function which Vercel treats as a serverless function. chat.ts sends a request to that function. That request carries a payload variable which contains the user's message, the conversation up to this point, and the user and echo id numbers (so that the message and response can be saved with the correct identification in supabase). The response to that request given by the generateReponse.ts script contains the McCon-Echo's response to the current user message twice (one to be saved to entered into the chat in response to the user and another to be added to the context window), the user's message, and the text content from Greenlights that was generated from the vector search, all of which is added to the current context window.

The responses are generated when generateResponse.ts makes an API call to Gemini's text generator with a set of instructions for how to answer along with the last two echanges worth of conversation and their corresponding Greenlights text support, a few shot prompt or two (few shot prompts are Q&A examples used for styling), and any past interactions that may have been relevant enough (i.e. surpassed some vector siilarity threshold) for McCon-echo to consider in his response. 

🚀 **Highlights and Acheivements**  
Matthew McCon-Echo acheives fantastic style, and decent factual accuracy utilizing Gemmini 2.0 for text generation. Gemini 2.0 is (roughly) 67% cheaper for text generation than Gemini 2.5 flash and 92% compared to Gemini 2.5 pro. These models, and even more recent models are what is generally reccomended for RAG systems that aim to replicate style, not just regurgitate facts.

McCon-Echo acheives this style via few-shot prompts and text from Greenlights that was not designed for the purpose it was utilized for. The business model this was intended to replicate involves 80,000 words worth of responses to carefully crafted interview questions designed to allow for the encoding of an individual's essence more feasible.

📊 **Instructions for Replicating this project**   
First, the Greenlights text must be chunked. To do this you run chunk_greenlights_text.py first, followed by gemini_embedding.py, followed by embeddings_to_supabase.py. You must also run the embedding scripts and embeddings to supabse for both the short few shot and long few shot exammples. However, order does not matter here. The required packages for the python scripts are contained in the requirements.txt file and can be installed with pip install -r requirements.txt.

The TypeScript and HTML files obviously do not need to be run like python scripts and are there to be interected with and triggered by user interaction. However, you will have to generate your own API keys for supabase, gemini, and vercel. You will also have to create the corresponding tables in Supabase for the sql functions to act on (these tables are where your vector embeddings are stored and retrieved from). LLMs are pretty efficient at walking you through this process, and Supabase even has it's own language model assistant present to help you with any confusion you may encounter on their platform.

📊 **Matthew McCon-Echo Limitations** 
The occasional context breaking and factual misrepresentations have already been noted. He is also a little slow on the first response because Vercel has to spin up a new container for the handler function on a cold start. Also, while I have incorporated stripe's payment system superficially, I have not set up the conditional that allows the user to continue interacting with McCon-echo after they have reached their five minute limit and then paid to continue. 

Finally, the end goal is to model McConaughey's voice and have McCon-Echo not only use McConaughey style phrases and responses but also McConaughey replicate his voice, inflection, and cadence. Greenlights has audio versions which I have already extraced. Coqui-TTS provides a package for trainign text to speech models like this. It involves training a model to make predictions about mel spectrograms based on text, and then training a second model to replicate McConaughey's voice based on the mel spectrograms. I simply have not implemented this yet.

⚙️ **Repository Structure**
```
⚙️ **Repository Structure**
echotree_lite/
├── api/ # Vercel serverless functions & backend scripts
│   ├── create-checkout-seesion.ts # Stripe checkout endpoint
│   ├── generateResponse.ts # Main RAG endpoint
│   ├── rag.py # Python script for retrieval-augmented generation
│   └── supabaseServer.ts # Supabase server-side helper functions
├── public/ # Static HTML pages
│   ├── cancel.html # Payment cancelled page
│   └── success.html # Payment success page
├── python_scripts/ # Python scripts for embeddings and chunking
│   ├── chunk_greenlights_text.py
│   ├── chunking_utilities.py
│   ├── embeddings_to_supabase.py
│   ├── few_shot_long_chunks.json
│   ├── few_shot_long_embeding.py
│   ├── few_shot_long_embeddings.json
│   ├── few_shot_long_embeddings_to_supabase.py
│   ├── few_shot_short_chunks.json
│   ├── few_shot_short_embedding.py
│   ├── few_shot_short_embeddings.json
│   ├── few_shot_short_embeddings_to_supabase.py
│   ├── gemini_embedding.py
│   ├── greenlights_chunks.json
│   ├── greenlights_embeddings.json
│   └── requirements.txt
├── src/ # Frontend TypeScript & CSS
│   ├── chat.ts
│   ├── counter.ts
│   ├── main.ts
│   ├── payment-plans.ts
│   ├── reset-password.ts
│   ├── set-new-password.ts
│   ├── style.css
│   ├── supabaseConfig.ts # Singleton Supabase client
│   ├── typescript.svg
│   └── vite-env.d.ts
├── .gitignore
├── chat.html
├── few_shots.txt
├── few-shot-long.txt
├── few-shot-short.txt
├── greenlights.txt
├── index.html
├── package.json
├── package-lock.json
├── payment-plans.html
├── reset-password.html
├── set-new-password.html
├── tsconfig.json
├── vite.config.ts
├── vite.svg
└── zero_shot_prompt.txt

```

1. Create and activate your environment for Python:

**Using Conda (recommended on Windows):**
```powershell
conda create --name project_name_env python=3.10
conda activate project_name_env
```

---

## 🧠 Authors

- Developed by Brett Kunkel | [www.linkedin.com/in/brett-kunkel](www.linkedin.com/in/brett-kunkel) | brttkunkel@gmail.com

---

## 📜 License

This project is licensed under the MIT License.
