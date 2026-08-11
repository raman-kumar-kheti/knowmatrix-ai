export default function Message({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-3xl rounded-xl px-4 py-3 text-sm ${
          isUser ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-800 border border-slate-200"
        }`}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
        {!isUser && message.citations?.length > 0 && (
          <div className="mt-2 pt-2 border-t border-slate-300 text-xs text-slate-600 space-y-1">
            <p className="font-semibold">Sources:</p>
            {message.citations.map((c, idx) => (
              <div key={idx}>
                [{idx + 1}] <span className="font-medium">{c.source}</span> (p.{c.page}, score: {c.score})
              </div>
            ))}
          </div>
        )}
        {!isUser && message.evaluation && (
          <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
            <span>Faithfulness: {message.evaluation.faithfulness_score}</span>
            <span>Relevance: {message.evaluation.answer_relevance}</span>
            <span>Risk: {message.evaluation.hallucination_risk}</span>
            <span>Latency: {message.latency_ms} ms</span>
          </div>
        )}
      </div>
    </div>
  );
}
