import { useEffect, useState } from "react";
import { deleteDocument, getMetrics, listDocuments } from "../api/client";
import ChatBox from "../components/ChatBox";
import Upload from "../components/Upload";

export default function Home() {
  const [docs, setDocs] = useState([]);
  const [metrics, setMetrics] = useState(null);

  const refresh = async () => {
    try {
      const [d, m] = await Promise.all([listDocuments(), getMetrics()]);
      setDocs(d);
      setMetrics(m);
    } catch {
      // Backend may still be starting
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleDelete = async (docId) => {
    await deleteDocument(docId);
    refresh();
  };

  return (
    <div className="min-h-screen bg-slate-100 py-6 px-4">
      <div className="max-w-5xl mx-auto space-y-5">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">KnowMatrix AI</h1>
            <p className="text-sm text-slate-600">Enterprise document intelligence and RAG</p>
          </div>
          {metrics && (
            <div className="flex gap-4 text-xs bg-white px-3 py-2 rounded-lg border border-slate-200">
              <span>Docs: <b>{metrics.indexed_documents}</b></span>
              <span>Queries: <b>{metrics.total_queries}</b></span>
              <span>Avg Faithfulness: <b>{metrics.avg_faithfulness_score}</b></span>
              <span>Avg Latency: <b>{metrics.avg_latency_ms} ms</b></span>
            </div>
          )}
        </header>

        <Upload onUploaded={refresh} />

        {docs.length > 0 && (
          <div className="p-4 bg-white rounded-xl shadow-sm border border-slate-200">
            <h2 className="text-sm font-semibold text-slate-700 mb-2">Indexed Documents</h2>
            <div className="space-y-1.5">
              {docs.map((doc) => (
                <div key={doc.doc_id} className="flex items-center justify-between text-xs bg-slate-50 px-3 py-2 rounded">
                  <span>
                    <b>{doc.filename}</b> · {doc.page_count}p · {doc.chunk_count} chunks · role: <code>{doc.access_role}</code>
                  </span>
                  <button
                    onClick={() => handleDelete(doc.doc_id)}
                    className="text-red-600 hover:underline"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <ChatBox onAnswered={refresh} />
      </div>
    </div>
  );
}
