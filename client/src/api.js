const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api";
const FILE_BASE_URL = API_BASE_URL.replace(/\/api$/, "");

async function parseResponse(response) {
  if (response.status === 204) {
    return null;
  }

  const rawBody = await response.text();
  let data = null;

  if (rawBody) {
    try {
      data = JSON.parse(rawBody);
    } catch {
      data = { message: rawBody };
    }
  }

  if (!response.ok) {
    throw new Error(data?.message || `Request failed with status ${response.status}.`);
  }

  return data;
}

export async function fetchAnalyses() {
  const response = await fetch(`${API_BASE_URL}/analyses`);
  return parseResponse(response);
}

export async function uploadAnalysis(formData) {
  const response = await fetch(`${API_BASE_URL}/analyses`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(response);
}

export async function removeAnalysis(id) {
  const response = await fetch(`${API_BASE_URL}/analyses/${id}`, {
    method: "DELETE",
  });
  return parseResponse(response);
}

export async function fetchAnalysisReport(id) {
  const response = await fetch(`${API_BASE_URL}/analyses/${id}/report`);
  return parseResponse(response);
}

export async function fetchTrainingReport() {
  const response = await fetch(`${API_BASE_URL}/training-report`);
  return parseResponse(response);
}

export function resolveImageUrl(imageUrl) {
  return `${FILE_BASE_URL}${imageUrl}`;
}
