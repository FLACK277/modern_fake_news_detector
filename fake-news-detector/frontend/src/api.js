const API_BASE = process.env.REACT_APP_API_BASE || "/api";

export async function predictNews(text) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    let message = "Prediction request failed";
    try {
      const data = await response.json();
      if (data?.detail) {
        message = typeof data.detail === "string" ? data.detail : message;
      }
    } catch {
      // Keep fallback message when response is not JSON.
    }
    throw new Error(message);
  }

  return response.json();
}
