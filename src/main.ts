import { supabase } from "./supabaseConfig";

const pathname = window.location.pathname;

// Load TypesCript scripts to their corresponding webpages
if (pathname.endsWith('chat.html')) {
  import('./chat');
} else if (pathname.endsWith('payment-plans.html')) {
  import('./payment-plans');
} else if (pathname.endsWith('reset-password.html')) {
  import('./reset-password');
} else if (pathname.endsWith('set-new-password.html')) {
  import('./set-new-password');
}

// Grab the input fields and button from the DOM
const emailInput = document.getElementById('email-input') as HTMLInputElement;
const passwordInput = document.getElementById('password-input') as HTMLInputElement;
const signInButton = document.getElementById('sign-in-button') as HTMLButtonElement;
const resetPasswordButton = document.getElementById('reset-password-button') as HTMLButtonElement;
const authMessage = document.getElementById('auth-message') as HTMLElement;

async function init() {
  await supabase.auth.signOut();
}

init();

// Navigate to reset-password.html when clicked
resetPasswordButton.addEventListener('click', () => {
  window.location.href = "reset-password.html"; // This will take the user to the new page
});

// Attach a click handler to the sign-in button
signInButton.addEventListener('click', async () => {
  const email = emailInput.value.trim();
  const password = passwordInput.value.trim();

  // Basic validation to ensure both fields are filled
  if (!email || !password) {
    authMessage.textContent = "Please enter both email and password.";
    return;
  }
  
  // We don't want the user to restart the whole function because they get impatient
  signInButton.disabled = true;

  console.log("Attempting sign-in with:", { email, password });

  // Attempt to sign in
  let { data: _data, error } = await supabase.auth.signInWithPassword({ email, password });

  // If user doesn't exist, automatically sign them up
  if (error) {
    const { data: signUpData, error: signUpError } = await supabase.auth.signUp({ email, password });
    if (signUpError) {
      console.error(signUpError.message);
      authMessage.textContent = "Sign-in failed.\n Please make sure you are entering a " +
                                "valid email address and a password at least 6 characters long\n";
      signInButton.disabled = false;
      return;
    }

    authMessage.style.color = "green";
    authMessage.textContent = 
    "Account created! A confirmation email will be sent to the email you provided.\n";

    // Log success for debugging
    console.log("New account created for:", signUpData.user?.email);

    // Delay redirect by 22 seconds (22000 ms)
    setTimeout(() => {
      window.location.href = "chat.html";
    }, 22000);
  }

  else {
    window.location.href = "chat.html";
  }
});
