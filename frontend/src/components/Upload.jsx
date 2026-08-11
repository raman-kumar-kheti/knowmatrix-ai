import { useState } from "react";
import { uploadDocument } from "../api/client";

export default function Upload({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [accessRole, setAccessRole] = useState("public");
  const [chunkSize, setChunkSize] = useState(800);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setStatusMsg("");
    try {
      const res = await uploadDocument(file, accessRole, chunkSize, chunkOverlap);
      setStatusMsg(`Indexed ${res.filename} (${res.chunk_count} chunks)`);
      setFile(null);
      if (onUploaded) onUploaded();
    } catch (err) {
      setStatusMsg(err.response?.data?.detail || "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-4 bg-white rounded-xl shadow-sm border border-slate-200 space-y-3">
      <h2 className="text-lg font-semibold text-slate-800">Upload Document (PDF / TXT / MD)</h2>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
        <input
          type="file"
          accept=".pdf,.txt,.md"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="text-sm text-slate-600"
        />
        <label className="text-xs text-slate-600">
          Access Role
          <select
            value={accessRole}
            onChange={(e) => setAccessRole(e.target.value)}
            className="mt-1 w-full border rounded-lg px-2 py-1.5 text-sm"
          >
            <option value="public">public</option>
            <option value="internal">internal</option>
            <option value="admin">admin</option>
          </select>
        </label>
        <label className="text-xs text-slate-600">
          Chunk Size / Overlap
          <div className="mt-1 flex gap-2">
            <input
              type="number"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
              className="w-1/2 border rounded-lg px-2 py-1.5 text-sm"
            />
            <input
              type="number"
              value={chunkOverlap}
              onChange={(e) => setChunkOverlap(Number(e.target.value))}
              className="w-1/2 border rounded-lg px-2 py-1.5 text-sm"
            />
          </div>
        </label>
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50"
        >
          {uploading ? "Indexing..." : "Upload & Index"}
        </button>
      </div>
      {statusMsg && <p className="text-sm text-indigo-700">{statusMsg}</p>}
    </div>
  );
}
