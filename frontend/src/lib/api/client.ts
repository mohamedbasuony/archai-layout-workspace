export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const direct = apiUrl(path);
  let res: Response;
  try {
    res = await fetch(direct, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    // Fallback to same-origin proxy route when direct backend URL fails (CORS/network).
    try {
      res = await fetch(`/api${path}`, init);
    } catch (proxyErr) {
      if (proxyErr instanceof DOMException && proxyErr.name === "AbortError") {
        throw proxyErr;
      }
      throw new ApiError(
        503,
        `Backend API unreachable at ${direct}. Start backend on 127.0.0.1:8000.`,
      );
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = body.detail || res.statusText;
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export function apiUrl(path: string): string {
  const envBase = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "");
  if (envBase) {
    return `${envBase}/api${path}`;
  }
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host === "127.0.0.1" || host === "localhost") {
      return `http://127.0.0.1:8000/api${path}`;
    }
  }
  return `/api${path}`;
}
