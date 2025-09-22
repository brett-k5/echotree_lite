async function goToCheckout(plan: string) {
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
}

document.getElementById("monthly-subscription")?.addEventListener("click", () => goToCheckout("monthly"));
document.getElementById("annual-subscription")?.addEventListener("click", () => goToCheckout("annual"));
document.getElementById("permanent-access")?.addEventListener("click", () => goToCheckout("permanent"));