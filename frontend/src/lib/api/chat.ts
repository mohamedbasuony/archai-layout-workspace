import { apiUrl } from "./client";

export interface ChatModelListResponse {
  models: string[];
  default_model: string;
  vision_models: string[];
  base_url: string;
}

export interface ChatMessagePayload {
  role: "user" | "assistant" | "system";
  content:
    | string
    | Array<
        | { type: "text"; text: string }
        | { type: "image_url"; image_url: { url: string } }
      >;
}

export interface ChatCompletionPayload {
  messages: ChatMessagePayload[];
  model?: string | null;
  temperature?: number;
  stream?: boolean;
  context?: Record<string, unknown>;
}

export interface ChatCompletionResult {
  text: string;
  model?: string;
}

interface StreamDelta {
  type: "delta";
  delta: string;
}

interface StreamDone {
  type: "done";
  text: string;
  model?: string;
}

interface StreamError {
  type: "error";
  error: string;
}

type StreamEvent = StreamDelta | StreamDone | StreamError;

function parseErrorText(status: number, raw: string): string {
  try {
    const parsed = JSON.parse(raw) as { detail?: string };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return `HTTP ${status}: ${parsed.detail}`;
    }
  } catch {
    // ignore json parse errors
  }
  return `HTTP ${status}: ${raw || "Request failed"}`;
}

export async function getChatModels(): Promise<ChatModelListResponse> {
  const res = await fetch(apiUrl("/chat/models"));
  if (!res.ok) {
    const raw = await res.text();
    throw new Error(parseErrorText(res.status, raw));
  }
  return (await res.json()) as ChatModelListResponse;
}

export async function createChatCompletion(
  payload: ChatCompletionPayload,
  onDelta?: (delta: string) => void,
): Promise<ChatCompletionResult> {
  const useStream = payload.stream !== false;
  const res = await fetch(apiUrl("/chat/completions"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: useStream }),
  });

  if (!res.ok) {
    const raw = await res.text();
    throw new Error(parseErrorText(res.status, raw));
  }

  if (!useStream) {
    const data = (await res.json()) as ChatCompletionResult;
    if (typeof data.text !== "string") {
      throw new Error("Invalid completion response.");
    }
    return data;
  }

  if (!res.body) {
    throw new Error("Streaming response body is missing.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let model: string | undefined;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    let splitIndex = buffer.indexOf("\n\n");
    while (splitIndex !== -1) {
      const chunk = buffer.slice(0, splitIndex);
      buffer = buffer.slice(splitIndex + 2);

      for (const rawLine of chunk.split("\n")) {
        const line = rawLine.trim();
        if (!line.startsWith("data:")) {
          continue;
        }
        const payloadText = line.slice(5).trim();
        if (!payloadText) {
          continue;
        }

        const event = JSON.parse(payloadText) as StreamEvent;
        if (event.type === "delta") {
          text += event.delta;
          onDelta?.(event.delta);
        } else if (event.type === "done") {
          text = event.text ?? text;
          model = event.model;
        } else if (event.type === "error") {
          throw new Error(event.error || "Streaming completion failed.");
        }
      }

      splitIndex = buffer.indexOf("\n\n");
    }
  }

  return { text, model };
}
