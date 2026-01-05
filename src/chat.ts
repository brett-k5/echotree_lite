// Import supabase client
import { GoogleGenerativeAI } from "@google/generative-ai";
import { loadStripe } from "@stripe/stripe-js";
import { supabase } from "./supabaseConfig";
import type { User } from '@supabase/supabase-js';

async function init() {
  await loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!);
}
 init()
const localUsage = new Map(); // stores how many seconds of voice each user has used
const usageLimitSeconds = 5 * 60; // 5 minutes limit per user (in seconds)
const speechRate = 0.75; // speed at which the Echo speaks
const averageWordsPerMinute = 160; // estimate for calculating how long speech takes
const genAI = new GoogleGenerativeAI(import.meta.env.VITE_GEMINI_API_KEY);

const signOutButton = document.getElementById("sign-out-button") as HTMLButtonElement;
const sendButton = document.getElementById("send-button") as HTMLButtonElement;
const subscriptionButton = document.getElementById("subscription-button") as HTMLButtonElement;
const messageElement = document.getElementById("usage-message") as HTMLDivElement; // HTML element to display warning messages
const input = document.getElementById('user-input') as HTMLInputElement;
const chatWindow = document.getElementById('chat-window') as HTMLElement;

sendButton.disabled = true;

// currentUser can either be a User or null (if no one is signed in) 
let currentUser: User | null = null;

supabase.auth.onAuthStateChange((_, session) => {
  if (session?.user) {
    currentUser = session.user;
    console.log("User signed in:", currentUser.email);
    sendButton.disabled = false;
  } 
  else {
    currentUser = null;
    window.location.href = '/login.html';
    sendButton.disabled = true;
  }
});


signOutButton.addEventListener("click", async () => {
  try {
    await supabase.auth.signOut();
    console.log("User signed out");
    window.location.href = "/login.html"; // redirect to login
  } catch (error) {
    console.error("Error signing out:", error);
  }
});

sendButton.addEventListener("click", sendMessage);


subscriptionButton.addEventListener("click", () => {
  window.location.href = "payment-plans.html";
});


interface currentExchange {
  previousUserMessage: string;
  previousEchoResponse: string;
  previousTextContent: string[];
  };

let currentConversation: currentExchange[] = []

// Define a function to inject echo response into the chat window
async function generateRAGResponse(userMessage: string, user_id: string, echo_id: string): Promise<string> {

  if (!user_id || !echo_id) {
  console.error("Cannot send message: missing user or echo ID");
  return;
  }

  const payload = {
  userMessage: userMessage, // from input.value.trim()
  currentConversation, // will be an empty array on the first call.
  user_id,
  echo_id
  };

  const res = await fetch('/api/generateResponse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ),
  });

  if (!res.ok) {
    console.error("Server error:", res.statusText);
    alert("Something went wrong. Please try again later.");
    return "Sorry, I couldn't generate a response just now.";
  }
  
  let previousUserMessage = "";
  let previousEchoResponse = "";
  let previousTextContent: string[] = [];
  let echoResponse: string = "";


  try {
    const {
      previousUserMessage: prevUserMess,
      previousEchoResponse: prevEchResp, 
      previousTextContent: prevTextCont,  
      echoResponse: echResp } = await res.json();
      
      previousUserMessage = prevUserMess
      previousEchoResponse = prevEchResp
      previousTextContent = prevTextCont
      echoResponse = echResp

    currentConversation.push({
      previousUserMessage,
      previousEchoResponse,
      previousTextContent,
    })
  }
 
  catch (err) {
    console.error("Failed to parse JSON:", err);
    alert("Server returned an invalid response.");
    return "Sorry, something went wrong while processing your message.";
  }
  
  return echoResponse;
}


async function getOrCreateEcho(userId: string, echoName: string) { 
  // Tries to find an existing Echo for this user
  let { data: echo, error: _error } = await supabase
    .from('echos')        // table name
    .select('*')          // get all columns
    .eq('name', echoName) // where name matches echoName
    .eq('user_id', userId) // where user_id matches the current user
    .single();            // expect only one row

  if (!echo) {
    // If no Echo exists, create a new one
    const { data, error: _error } = await supabase 
      .from('echos')
      .insert([{ user_id: userId, name: echoName }]) // insert a row with user_id and name
      .select() 
      .single();
    return data; // return the newly created Echo
  }

  return echo; // return existing Echo if found
}


function updateUsageBar(echoId: string) {
  // assign the value for the given echoID if it is not 0 to usedSeconds variable. If the value for echoID is a falsy value,
  const usedSeconds = localUsage.get(echoId) || 0; //  assign value of 0 to usedSeconds variable
  const usedMinutes = Math.floor(usedSeconds / 60); // assign number of inutes used in integer floors to usedMinutes variable
  document.getElementById("used-minutes")!.textContent = usedMinutes.toString(); // fill bar with number of minutes used - floor values used
}


async function initializeUsage(echoId: string) {
  const { data: echo, error } = await supabase
    .from('echos')
    .select('usage_seconds')
    .eq('id', echoId)
    .single();

  if (!error && echo) {
    localUsage.set(echoId, echo.usage_seconds || 0);
    updateUsageBar(echoId); // update UI immediately
  }

  if (error) {
    console.error("Failed to fetch usage:", error);
    return;
}
}


async function speakText(text: string, echoId: string, playButton: HTMLButtonElement) {
  if (!echoId) {
    console.error("No echoId provided");
    return;
  }

  // Initialize local usage if needed
  if (!localUsage.has(echoId)) localUsage.set(echoId, 0);

  const usedSeconds = localUsage.get(echoId);

  // Check quota locally first
  if (usedSeconds >= usageLimitSeconds) {
    if (!currentEcho) return; // early exit
    alert(`You have reached your usage limit for ${currentEcho!.name} Please review payment options by clicking the "Subscription Options" button below`)
    sendButton.disabled = true;
    playButton.disabled = true;
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = speechRate;

  const words = text.trim().split(/\s+/).length;
  const estimatedSeconds = (words / averageWordsPerMinute) * 60;

  utterance.onend = async () => {
    // Update local cache
    const newUsage = Math.min(localUsage.get(echoId) + estimatedSeconds, usageLimitSeconds);
    localUsage.set(echoId, newUsage);
    updateUsageBar(echoId);

    // Show warnings if necessary
    const remainingSeconds = usageLimitSeconds - newUsage;
    if (remainingSeconds === 120) {
      messageElement.textContent = "You only have about two more minutes of usage!";
    }
    else if (remainingSeconds <= 90) {
      messageElement.textContent = "" 
    }
    

    // Async sync to Supabase
    try {
      await supabase
        .from('echos')
        .update({ usage_seconds: newUsage })
        .eq('id', echoId);
    } catch (err) {
      console.error("Failed to sync usage:", err);
      // Optional: retry later or queue updates
    }
  };

  speechSynthesis.speak(utterance);
}


function extractEchoNameFromMessage(message: string) {
  const words = message.trim().split(/\s+/);
  // Assume the first word is just a greeting, so drop it
  return words.slice(1).join(" ");
}

// current user is given by supabase authentication
// echoId is given by first message

// If you have an interface for Echo:
interface Echo {
  id: string;       // UUID as string
  name: string;
  usage_seconds?: number;
}

// Then declare currentEcho like this:
let currentEcho: Echo | null = null;

async function sendMessage() {
  // Use cached user instead of calling getUser() every time
  if (!currentUser) return alert("Please sign in first!");

  // Get the cached user's ID
  const userId = currentUser!.id;

  const message = input.value.trim();
  if (!message) return;
  
  if (!currentEcho) {
    // Only create/fetch the echo the first time
    currentEcho = await getOrCreateEcho(userId, extractEchoNameFromMessage(message));
    document.getElementById("echo-header")!.textContent = `Echo Profile: ${currentEcho!.name}`;
    await initializeUsage(currentEcho!.id);
  }

  const usedSeconds = localUsage.get(currentEcho!.id) || 0;
  if (usedSeconds >= usageLimitSeconds) {
    alert(`You have reached your usage limit for ${currentEcho!.name}. Please review payment options by clicking the "Subscription Options" button below.`);
    sendButton.disabled = true;
    return;
  }

  // Display user message in chat
  const userMsg = document.createElement('div');
  userMsg.textContent = "You: " + message;
  chatWindow.appendChild(userMsg);

  const response = await generateRAGResponse(message, userId, currentEcho!.id)

  // Display and speak Echo response
  const echoContainer = document.createElement('div');
  const responseText = document.createElement('span');
  responseText.textContent = `${currentEcho!.name}: ${response}`;
  echoContainer.appendChild(responseText);

  const playButton = document.createElement('button');
  playButton.textContent = '🔊';
  playButton.style.marginLeft = '10px';
  playButton.onclick = () => speakText(response, currentEcho!.id, playButton);
  echoContainer.appendChild(playButton);

  chatWindow.appendChild(echoContainer);

  // Insert echo response into echo_responses table
  await supabase.from('echo_responses').insert([{
    echo_id: currentEcho!.id,  
    user_id: userId,        
    echo_response: response,         
    user_message: message
  }]);

  // Speak and track usage
  speakText(response, currentEcho!.id, playButton);

  input.value = ''; // clear input
}