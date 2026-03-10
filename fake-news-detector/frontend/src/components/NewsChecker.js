import { useMemo, useState } from "react";
import { predictNews } from "../api";

function NewsChecker() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const isTooShort = text.trim().length < 10;

  const badgeStyle = useMemo(() => {
    if (!result) return "bg-slate-100 text-slate-700";
    return result.prediction === "Real"
      ? "bg-green-100 text-green-700"
      : "bg-red-100 text-red-700";
  }, [result]);

  const ringStyle = useMemo(() => {
    if (!result) return "ring-slate-200";
    return result.prediction === "Real" ? "ring-green-400" : "ring-red-400";
  }, [result]);

  const onSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);

    if (isTooShort) {
      setError("Please enter at least 10 characters.");
      return;
    }

    setLoading(true);
    try {
      const prediction = await predictNews(text.trim());
      setResult(prediction);
    } catch (err) {
      setError(err.message || "Something went wrong while checking this news.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl rounded-2xl bg-white/80 p-6 shadow-xl backdrop-blur-sm md:p-8">
      <form onSubmit={onSubmit} className="space-y-4">
        <label htmlFor="news-text" className="block text-sm font-semibold text-slate-700">
          Paste news content
        </label>

        <textarea
          id="news-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Paste a news article, post, or claim here..."
          className="min-h-48 w-full rounded-xl border border-slate-300 bg-white p-4 text-slate-800 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
        />

        <button
          type="submit"
          disabled={loading}
          className="inline-flex items-center justify-center rounded-xl bg-sky-600 px-6 py-3 font-semibold text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-sky-300"
        >
          {loading ? (
            <span className="inline-flex items-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/50 border-t-white" />
              Checking...
            </span>
          ) : (
            "Check News"
          )}
        </button>
      </form>

      {error && (
        <div className="mt-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {result && (
        <div className={`mt-6 rounded-xl bg-white p-5 ring-2 ${ringStyle}`}>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-medium text-slate-500">Prediction</p>
              <p className="mt-1 text-2xl font-bold text-slate-900">{result.prediction}</p>
            </div>
            <span className={`rounded-full px-3 py-1 text-sm font-semibold ${badgeStyle}`}>
              {result.prediction === "Real" ? "Likely trustworthy" : "Likely misleading"}
            </span>
          </div>

          <div className="mt-4">
            <p className="text-sm font-medium text-slate-500">Confidence</p>
            <p className="mt-1 text-xl font-semibold text-slate-900">
              {(result.confidence * 100).toFixed(2)}%
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default NewsChecker;
