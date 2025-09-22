//   Moved function definition to top for clarity
async function goToCheckout(plan: string) {
  try {
    const response = await fetch("/api/create-checkout-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan }), // pass the chosen plan to backend
    });

    if (!response.ok) {
      const error = await response.json();
      alert(`Error: ${error.error}`);
      return;
    }

    const { url } = await response.json();
    window.location.href = url; // redirect to Stripe checkout
  } catch (err) {
    //   Catch unexpected network or runtime errors
    console.error("Checkout request failed:", err);
    alert("Something went wrong while initiating checkout. Please try again.");
  }
}

//   Added null-check logging for debugging
const monthlyBtn = document.getElementById("monthly-subscription");
const annualBtn = document.getElementById("annual-subscription");
const permanentBtn = document.getElementById("permanent-access");

if (!monthlyBtn || !annualBtn || !permanentBtn) {
  console.warn("One or more subscription buttons are missing from the DOM."); //   Helps catch missing elements
}

monthlyBtn?.addEventListener("click", () => goToCheckout("monthly"));
annualBtn?.addEventListener("click", () => goToCheckout("annual"));
permanentBtn?.addEventListener("click", () => goToCheckout("permanent"));