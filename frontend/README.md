# ArchAI Frontend

This Next.js application provides the document workspace for the ArchAI backend.

## Development

```bash
npm install
npm run dev -- --hostname 127.0.0.1 --port 3001
```

Open [http://127.0.0.1:3001/workspace](http://127.0.0.1:3001/workspace).

By default, local API requests use `http://127.0.0.1:8000/api`. Set
`NEXT_PUBLIC_BACKEND_URL` to point at a different backend:

```bash
NEXT_PUBLIC_BACKEND_URL=https://example.com npm run dev
```

## Checks

```bash
npm run lint
npm run build
```
