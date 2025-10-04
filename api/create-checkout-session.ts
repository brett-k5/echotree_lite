// api/create-checkout-session.ts

import Stripe from 'stripe';
//   Stripe import remains the same

//   Stripe initialization with secret key and API version
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2023-08-16',
});

//   Removed: `import type { VercelRequest, VercelResponse } from '@vercel/node`
//   Replaced with native Node types for compatibility without @vercel/node
import type { IncomingMessage, ServerResponse } from 'http';

//   Replaced Vercel-specific types with extended native types
export default async function handler(
  req: IncomingMessage & { method?: string; body?: any },
  res: ServerResponse & { status: (code: number) => any; json: (body: any) => void }
) {
  //   Restrict to POST requests only
  if (req.method !== "POST") {
    return res.status(405).json({ 
      error: `For security reasons this page cannot be accessed directly (this includes reloading the page). 
              To view your subscription options again, please log back into EchoTree Lite and click the "Subscription Options"
              button on the chat page.` // Minor punctuation fix for clarity
    });
  }

  //   Extract plan from request body
  const { plan } = req.body;

  //   Map user selection to Stripe Price IDs
  const priceMap: Record<string, string> = {
    monthly: "price_1S9HlFChWty64KkA2013c3No",
    annual: "price_1S9Hm5ChWty64KkA0OFO2Ftv",
    permanent: "price_1S9HnGChWty64KkACLVyfFJR",
  };

  //   Validate plan
  if (!priceMap[plan]) {
    return res.status(400).json({ error: "Invalid subscription plan" });
  }

  try {
    //   Stripe session creation logic
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ["card"],
      mode: plan === "permanent" ? "payment" : "subscription",
      line_items: [{ price: priceMap[plan], quantity: 1 }],
      success_url: "https://echotree-lite.vercel.app/success.html",
      cancel_url: "https://echotree-lite.vercel.app/cancel.html",
      customer_creation: plan === "permanent" ? "always" : undefined // Improves subscription tracking
    });

    // Return session URL to frontend
    res.status(200).json({ url: session.url });
  } catch (err) {
    //   Error logging and response
    console.error("Stripe session creation failed:", err);
    res.status(500).json({ error: (err as Error).message });
  }
}