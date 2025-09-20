import { supabase } from "./supabaseConfig.ts";

// Grab DOM elements
const newPasswordInput = document.getElementById('new-password-input') as HTMLInputElement;
const confirmPasswordInput = document.getElementById('confirm-password-input') as HTMLInputElement;
const setPasswordButton = document.getElementById('set-password-button') as HTMLButtonElement;
const resetMessage = document.getElementById('reset-message') as HTMLElement;
const resetError = document.getElementById('reset-error') as HTMLElement;

setPasswordButton.addEventListener('click', async () => {
  const newPassword = newPasswordInput.value.trim();
  const confirmPassword = confirmPasswordInput.value.trim();

  resetError.textContent = "";
  resetMessage.textContent = "";

  // Basic validation
  if (!newPassword || !confirmPassword) {
    resetError.textContent = "Please fill in both fields.";
    return;
  }

  if (newPassword !== confirmPassword) {
    resetError.textContent = "Passwords do not match.";
    return;
  }

  setPasswordButton.disabled = true;
  const originalText = setPasswordButton.textContent;
  setPasswordButton.textContent = "Updating...";

  try {
    const { data, error } = await supabase.auth.updateUser({
      password: newPassword
    });

    if (error) {
      resetError.textContent = error.message;
      return;
    }

    resetMessage.textContent = "Your password has been successfully updated! You can now log in. " +
                               "You will be re-directed to the login page in a few seconds";
    newPasswordInput.value = "";
    confirmPasswordInput.value = "";

    // Optional: redirect to login after a short delay
    setTimeout(() => {
      window.location.href = "/index.html";
    }, 8000);

  } catch (err) {
    console.error("Unexpected error:", err);
    resetError.textContent = "An unexpected error occurred. Please try again later.";
  } finally {
    setPasswordButton.disabled = false;
    setPasswordButton.textContent = originalText;
  }
});