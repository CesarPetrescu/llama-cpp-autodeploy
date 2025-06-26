#!/usr/bin/env python3
"""Generic reranker using HuggingFace Transformers.

The script can score a list of documents for a query or run a minimal
HTTP API compatible with the llama.cpp ``/rerank`` endpoint.
"""

import argparse
import json
from typing import List
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(repo_or_path: str):
    """Load tokenizer and model from a repo or local path."""
    tokenizer = AutoTokenizer.from_pretrained(repo_or_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_or_path, trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


def rerank(query: str, docs: List[str], tokenizer, model, device: str = "cpu") -> List[tuple[str, float]]:
    pairs = [f"{query}\n{doc}" for doc in docs]
    encoded = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**encoded).logits[:, 0]
    scores = logits.cpu().tolist()
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank documents with Transformers")
    parser.add_argument("model", help="model repo or local path")
    parser.add_argument("query", nargs="?", help="query text")
    parser.add_argument("documents", nargs="*", help="documents to rank")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    parser.add_argument("--serve", action="store_true", help="run an HTTP server instead of CLI output")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    model.to(args.device)

    if args.serve or not args.query:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/rerank":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                query = data.get("query", "")
                docs = data.get("documents", [])
                top_n = data.get("top_n", len(docs))
                ranked = rerank(query, docs, tokenizer, model, args.device)[:top_n]
                resp = {
                    "results": [
                        {"document": d, "score": s} for d, s in ranked
                    ]
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())

        server = HTTPServer((args.host, args.port), Handler)
        print(f"Reranker serving on {args.host}:{args.port}")
        server.serve_forever()
    else:
        ranked = rerank(args.query, args.documents, tokenizer, model, args.device)
        for idx, (doc, score) in enumerate(ranked, 1):
            print(f"{idx}. {doc} (score={score:.4f})")


if __name__ == "__main__":
    main()
