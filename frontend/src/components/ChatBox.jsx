import { useState } from "react";
import { sendChat } from "../api/client";
import Message from "./Message";

export default function ChatBox({ onAnswered }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [userRole, setUserRole] = useState("public");
  const [topK, setTopK] = useState(4);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setQuestion("");
    setLoading(true);
    try {
      const res = await sendChat({
        question: trimmed,
        session_id: "web-session",
        user_role: userRole,
        top_k: topK,
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          citations: res.citations,
          evaluation: res.evaluation,
          latency_ms: res.latency_ms,
        },
      ]);
      if (onAnswered) onAnswered();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: err.response?.data?.detail || "Error querying assistant." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-800">Knowledge Assistant</h2>
        <div className="flex items-center gap-3 text-xs text-slate-600">
          <label>
            Role:
            <select
              value={userRole}
              onChange={(e) => setUserRole(e.target.value)}
              className="ml-1 border rounded px-2 py-1"
            >
              <option value="public">public</option>
              <option value="internal">internal</option>
              <option value="admin">admin</option>
            </select>
          </label>
          <label>
            Top-K:
            <input
              type="number"
              min={1}
              max={12}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="ml-1 w-14 border rounded px-2 py-1"
            />
          </label>
        </div>
      </div>

      <div className="h-96 overflow-y-auto space-y-3 p-2 bg-slate-50 rounded-lg border border-slate-100">
        {messages.length === 0 && (
          <p className="text-sm text-slate-400 text-center mt-16">
            Upload a document and ask a question to receive cited answers.
          </p>
        )}
        {messages.map((m, idx) => (
          <Message key={idx} message={m} />
        ))}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask a question about your indexed documents..."
          className="flex-1 border rounded-lg px-3 py-2 text-sm"
        />
        <button
          onClick={handleSend}
          disabled={loading || !question.trim()}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50"
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>
    </div>
  );
}
