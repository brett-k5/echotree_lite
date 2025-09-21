import { supabase } from "./supabaseConfig";

// Grab DOM elements
const emailInput = document.getElementById('reset-email-input') as HTMLInputElement;
const sendButton = document.getElementById('send-reset-link-button') as HTMLButtonElement;
const resetMessage = document.getElementById('reset-message') as HTMLElement;
const resetError = document.getElementById('reset-error') as HTMLElement;

// Handle sending password reset email
sendButton.addEventListener('click', async () => {
  const email = emailInput.value.trim();

  // Basic validation
  if (!email) {
    resetError.textContent = "Please enter your email address.";
    resetMessage.textContent = "";
    return;
  }

  // Disable button while processing
  sendButton.disabled = true;
  const originalText = sendButton.textContent;
  sendButton.textContent = "Sending...";

  resetError.textContent = "";
  resetMessage.textContent = "";

  try {
    const { data: _data, error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: window.location.origin + "/set-new-password.html", // optional: where users are sent after clicking the reset link
    });

    if (error) {
      resetError.textContent = "Please make sure you are entering the email address you signed up with.";
      sendButton.disabled = false;
      return;
    }

    // Success
    resetMessage.textContent = "A password reset link has been sent to your email. " +
                               "Check your inbox and follow the instructions to reset your password.";
    emailInput.value = "";

  } catch (err) {
    console.error("Unexpected error:", err);
    resetError.textContent = "An unexpected error occurred. Please try again later.";
  } finally {
    // Re-enable button and restore original text
    sendButton.disabled = false;
    sendButton.textContent = originalText;
  }
});
