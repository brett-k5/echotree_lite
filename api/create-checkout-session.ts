// api/create-checkout-session.ts
import { NextApiRequest, NextApiResponse } from "next";
import Stripe from "stripe";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: "2025-01-27",
});

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ 
        error: `For security reasons this page cannot be accessed directly (this includes reloading the page). 
                To view your subscription options again, please log back into echotree lite and click the "Subscription Options"
                button on the chat page`});
  }

  const { plan } = req.body;
  
  // Map user selection to Stripe Price IDs
  const priceMap: Record<string, string> = {
    monthly: "price_1S9HlFChWty64KkA2013c3No",
    annual: "price_1S9Hm5ChWty64KkA0OFO2Ftv",
    permanent: "price_1S9HnGChWty64KkACLVyfFJR",
  };

  if (!priceMap[plan]) {
    return res.status(400).json({ error: "Invalid subscription plan" });
  }


  try {
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ["card"],
      mode: plan === "permanent" ? "payment" : "subscription",
      line_items: [
        { price: priceMap[plan], quantity: 1 } // replace with your Stripe price ID
      ],
      success_url: "https://echotree-lite.vercel.app/success",
      cancel_url: "https://echotree-lite.vercel.app/cancel",
    });

    res.json({ url: session.url });
  } 
  catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
}