import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export const uploadDocument = async (file, accessRole = "public", chunkSize = 800, chunkOverlap = 150) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("access_role", accessRole);
  formData.append("chunk_size", String(chunkSize));
  formData.append("chunk_overlap", String(chunkOverlap));
  const res = await apiClient.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const listDocuments = async () => (await apiClient.get("/documents")).data;
export const deleteDocument = async (docId) => (await apiClient.delete(`/documents/${docId}`)).data;
export const getMetrics = async () => (await apiClient.get("/metrics")).data;
export const sendChat = async (payload) => (await apiClient.post("/chat", payload)).data;

export default apiClient;
