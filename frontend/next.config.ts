import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Expose ARCHAI_DEBUG_RAG to the client bundle so a single env var
  // enables both backend debug endpoints AND the frontend RAG report.
  env: {
    NEXT_PUBLIC_ARCHAI_DEBUG_RAG:
      process.env.NEXT_PUBLIC_ARCHAI_DEBUG_RAG ||
      process.env.ARCHAI_DEBUG_RAG ||
      "",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
