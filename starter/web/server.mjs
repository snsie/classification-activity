// Serve the Node app and the exported results: npm start (or node web/server.mjs [port])
// No npm dependencies. Serves the files in web/ and the JSON files in results/; it never retrains a model.
import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { gzipSync } from "node:zlib";

const WEB = new URL(".", import.meta.url).pathname;
const RESULTS = new URL("../results/", import.meta.url).pathname;
const PORT = Number(process.argv[2] || process.env.PORT || 8766);
// A host such as Render sets PORT and needs the app on every interface; on your computer it stays private.
const HOST = process.env.HOST || (process.env.PORT ? "0.0.0.0" : "127.0.0.1");
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml",
                ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon", ".woff2": "font/woff2" };

function send(req, res, file) {
  let body = readFileSync(file);
  const headers = { "Content-Type": TYPES[extname(file)], "Cache-Control": "no-cache" };
  if (/\bgzip\b/.test(req.headers["accept-encoding"] || "")) {
    body = gzipSync(body);
    headers["Content-Encoding"] = "gzip";
  }
  res.writeHead(200, headers).end(body);
}

createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^\/+/, "") || "index.html";
  const result = path.match(/^results\/(\w+)\.json$/);
  if (result) {
    const file = join(RESULTS, `${result[1]}.json`);
    return existsSync(file) ? send(req, res, file) : res.writeHead(404).end("No results yet: run python -m analysis");
  }
  const file = join(WEB, path);
  if (!path.startsWith("..") && path !== "server.mjs" && TYPES[extname(path)] && existsSync(file) && statSync(file).isFile()) {
    return send(req, res, file);
  }
  res.writeHead(404, { "Content-Type": "text/plain" }).end("Not found");
}).listen(PORT, HOST, () => console.log(`Judge the Model: http://localhost:${PORT}/`));
