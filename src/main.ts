import './style.css';

import { supabase } from "./supabaseConfig.ts";

// Grab the input fields and button from the DOM
const emailInput = document.getElementById('email-input') as HTMLInputElement;
const passwordInput = document.getElementById('password-input') as HTMLInputElement;
const signInButton = document.getElementById('sign-in-button') as HTMLButtonElement;
const authMessage = document.getElementById('auth-message')as HTMLElement;

await supabase.auth.signOut();

// Attach a click handler to the sign-in button
signInButton.addEventListener('click', async () => {
  const email = emailInput.value.trim();
  const password = passwordInput.value.trim();

  // Basic validation to ensure both fields are filled
  if (!email || !password) {
    authMessage.textContent = "Please enter both email and password.";
    return;
  }

  console.log("Attempting sign-in with:", { email, password });

  // Attempt to sign in
  let { data, error } = await supabase.auth.signInWithPassword({ email, password });

  // If user doesn't exist, automatically sign them up
  if (error) {
    const { data: signUpData, error: signUpError } = await supabase.auth.signUp({ email, password });
    if (signUpError) {
      console.error(signUpError.message);
      authMessage.textContent = "Sign-in failed.\n Please make sure you are entering a " +
                                "valid email address and a password at least 6 characters long\n";
      return;
    }
  

    // Log success for debugging
    console.log("New account created for:", signUpData.user?.email);
    window.location.href = "chat.html";
    return; // stop here — don’t auto-redirect
  }
  window.location.href = "chat.html";
});