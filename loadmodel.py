#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
loadmodel.py
------------
Tiny launcher for:
  1) llama.cpp llama-server (LLM or Embeddings) with Hugging Face GGUF download
  2) Transformers-based reranker (Qwen/Qwen3-Reranker-8B, 8-bit/4-bit with fallback)

Endpoints we expose:
  - llama.cpp LLM:         POST /completion
  - llama.cpp Embeddings:  POST /v1/embeddings
  - Reranker (Transformers):
      * POST /v1/embeddings  -> "embedding": [score]   (score \in [0..1], one scalar per pair)
      * POST /rerank         -> { "results": [ {index, score, document?}, ... ] }

Notes (reranker):
  - Pairs are built in Qwen chat format from (instruction, query, doc).
  - /v1/embeddings also accepts lines like:
        "query: <q>\ndocument: <d>"
    and returns an "embedding" with a single score.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Tuple

import memory_utils
from process_utils import register_process, terminate_process, unregister_process

# Optional deps:
try:
    from huggingface_hub import list_repo_files, hf_hub_download
    HF_OK = True
except Exception:
    HF_OK = False

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
LLAMA_SERVER = ROOT / "bin" / "llama-server"

# --------------------------- Utilities ---------------------------

def info(msg: str) -> None:
    print(msg, flush=True)

def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def run_capture(cmd: List[str], env: Optional[dict] = None) -> str:
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, env=env)
        return out
    except subprocess.CalledProcessError as e:
        output = e.output or ""
        msg = f"Command failed with exit code {e.returncode}: {cmd_str}"
        if output:
            msg += f"\nOutput:\n{output}"
        raise RuntimeError(msg) from e
    except Exception as e:
        raise RuntimeError(f"Failed to run command: {cmd_str}\nError: {e}") from e

def detect_ubatch_flag() -> Tuple[Optional[str], Optional[str]]:
    """Return ('--ubatch' or '--n-ubatch' if supported, None otherwise, and the alternate synonym)."""
    if not LLAMA_SERVER.exists():
        return (None, None)
    try:
        help_out = run_capture([str(LLAMA_SERVER), "--help"])
    except Exception as e:
        print(
            f"[WARN] Failed to detect ubatch flags via `llama-server --help`: {e}",
            file=sys.stderr,
        )
        return (None, None)
    have_ubatch = "--ubatch" in help_out
    have_nubatch = "--n-ubatch" in help_out
    primary = "--ubatch" if have_ubatch else ("--n-ubatch" if have_nubatch else None)
    secondary = "--n-ubatch" if primary == "--ubatch" else ("--ubatch" if primary == "--n-ubatch" else None)
    return (primary, secondary)


def ensure_moe_flags_available(want_moe: bool) -> None:
    """Fail fast if the llama-server build is too old to understand MoE offload flags."""
    if not want_moe:
        return
    if not LLAMA_SERVER.exists():
        die(
            "llama-server not found. Rebuild llama.cpp (e.g., `python autodevops.py --ref latest --now`) "
            "to enable Mixture-of-Experts offloading."
        )

    try:
        help_out = run_capture([str(LLAMA_SERVER), "--help"])
    except Exception as e:
        die(
            "Failed to inspect llama-server for MoE support via `--help`. "
            "Ensure llama-server is executable and rebuilt if necessary. "
            f"Details: {e}"
        )
    if "--cpu-moe" not in help_out and "--n-cpu-moe" not in help_out:
        die(
            "This llama-server build does not recognize --cpu-moe/--n-cpu-moe. "
            "Rebuild llama.cpp with a recent commit (e.g., `python autodevops.py --ref latest --now`)."
        )

def parse_ollama_ref(spec: str) -> Tuple[str, Optional[str]]:
    # org/repo[:quant_or_filename]
    if ":" in spec:
        repo, quant = spec.split(":", 1)
        return repo.strip(), quant.strip()
    return spec.strip(), None

def choose_gguf(files: List[str], want: Optional[str]) -> str:
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        die("No .gguf files found in repo.")

    if want:
        wl = want.lower()
        # exact file name
        for f in ggufs:
            if f.lower() == wl:
                return f
        # token/boundary match
        for f in ggufs:
            if re.search(rf"(^|[-_.]){re.escape(wl)}($|[-_.])", f.lower()):
                return f
        # substring fallback
        for f in ggufs:
            if wl in f.lower():
                return f

    # Default preference if none given
    rank = ["q8_0", "q6_k", "q6", "q5_k", "q5", "q4_k", "q4"]
    def score(name: str) -> int:
        n = name.lower()
        for i, k in enumerate(rank):
            if k in n:
                return 1000 - i
        return 0
    return sorted(ggufs, key=score, reverse=True)[0]

def resolve_gguf(spec: str, models_dir: Path, hf_token: Optional[str]) -> Path:
    """Return local path to a GGUF file, downloading from HF if needed."""
    p = Path(spec)
    models_dir.mkdir(parents=True, exist_ok=True)

    if p.exists():
        if p.is_dir():
            die(f"Path points to a directory, not a .gguf: {p}")
        if p.suffix.lower() != ".gguf":
            die(f"Expected a .gguf file, got: {p}")
        return p.resolve()

    if not HF_OK:
        die("huggingface_hub not installed. pip install huggingface-hub")

    repo, want = parse_ollama_ref(spec)
    try:
        files = list_repo_files(repo)
    except Exception as e:
        die(f"Failed to list files for {repo}: {e}")

    pick = choose_gguf(files, want)
    info(f"[hf] {repo} -> {pick}")
    try:
        local = hf_hub_download(
            repo_id=repo,
            filename=pick,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
            force_download=False,
            resume_download=True,
        )
        return Path(local).resolve()
    except Exception as e:
        die(f"Download failed for {repo}/{pick}: {e}")
    return Path()

def parse_max_memory_arg(s: Optional[str]) -> Optional[dict]:
    """
    Accepts:
      - "22GiB,22GiB,cpu=128GiB"              -> {0: "22GiB", 1: "22GiB", "cpu": "128GiB"}
      - "0=22GiB,1=22GiB,cpu=128GiB"          -> {0: "22GiB", 1: "22GiB", "cpu": "128GiB"}
      - "cuda:0=22GiB,gpu:1=22GiB,cpu=128GiB" -> {0: "22GiB", 1: "22GiB", "cpu": "128GiB"}
    """
    if not s:
        return None
    mm: dict = {}
    idx = 0
    for part in (x.strip() for x in s.split(",") if x.strip()):
        if "=" in part:
            dev, mem = part.split("=", 1)
            key = dev.strip().lower()
            for prefix in ("cuda:", "gpu:"):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key.isdigit():
                k = int(key)
            elif key in ("cpu", "mps", "disk"):
                k = key
            else:
                try:
                    k = int(key)
                except ValueError:
                    k = key
            mm[k] = mem.strip()
        else:
            mm[idx] = part
            idx += 1
    return mm

def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)

# --------------------------- GGUF: LLM / Embeddings ---------------------------

def launch_llama_server(
    model_path: Path,
    host: str,
    port: int,
    extra: List[str],
    n_gpu_layers: Optional[int],
    tensor_split: Optional[str],
    split_mode: Optional[str],
    ctx_size: Optional[int],
    n_cpu_moe: Optional[int],
    cpu_moe: bool,
    mmproj: Optional[str] = None,
    jinja: bool = False,
    reasoning_format: Optional[str] = None,
    no_context_shift: bool = False,
) -> subprocess.Popen:
    cmd = [
        str(LLAMA_SERVER), "--model", str(model_path),
        "--host", host, "--port", str(port),
    ]
    if n_gpu_layers is not None and n_gpu_layers > 0:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
    if split_mode:
        cmd += ["--split-mode", split_mode]
    if tensor_split:
        cmd += ["--tensor-split", tensor_split]
    if ctx_size:
        cmd += ["--ctx-size", str(ctx_size)]
    if cpu_moe:
        cmd.append("--cpu-moe")
    elif n_cpu_moe is not None and n_cpu_moe > 0:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]
    if mmproj:
        cmd += ["--mmproj", str(mmproj)]
    if jinja:
        cmd.append("--jinja")
    if reasoning_format:
        cmd += ["--reasoning-format", reasoning_format]
    if no_context_shift:
        cmd.append("--no-context-shift")
    if extra:
        # map --ubatch <-> --n-ubatch to what your llama-server supports
        primary, _secondary = detect_ubatch_flag()
        cur = []
        i = 0
        while i < len(extra):
            tok = extra[i]
            if tok in ("--ubatch", "--n-ubatch"):
                val = extra[i+1] if i+1 < len(extra) and not str(extra[i+1]).startswith("--") else None
                if primary is None:
                    i += 2 if val else 1
                    continue
                cur.append(primary)
                if val: cur.append(val)
                i += 2 if val else 1
                continue
            cur.append(tok); i += 1
        cmd += cur

    info(f"[llama] {shell_join(cmd)}")
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env)
    return register_process(proc)

def _wait_for_listen(host: str, port: int, timeout: float = 60.0) -> bool:
    import socket, time
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect((connect_host, port))
            s.close()
            return True
        except Exception:
            time.sleep(0.5)
    return False

def launch_llama_server_with_backoff(
    model_path: Path,
    host: str,
    port: int,
    extra: List[str],
    n_gpu_layers: Optional[int],
    tensor_split: Optional[str],
    split_mode: Optional[str],
    ctx_size: Optional[int],
    n_cpu_moe: Optional[int],
    cpu_moe: bool,
    mmproj: Optional[str] = None,
    jinja: bool = False,
    reasoning_format: Optional[str] = None,
    no_context_shift: bool = False,
) -> subprocess.Popen:
    """Start llama.cpp. If it fails (e.g., CUDA OOM), retry with fewer GPU layers so the rest stays on CPU RAM."""
    ensure_moe_flags_available(cpu_moe or (n_cpu_moe is not None and n_cpu_moe > 0))
    ngl_try = n_gpu_layers if (n_gpu_layers is not None and n_gpu_layers >= 0) else 999
    attempt = 0
    while True:
        attempt += 1
        info(f"[llama][try {attempt}] --n-gpu-layers={ngl_try}")
        proc = launch_llama_server(
            model_path,
            host,
            port,
            extra,
            ngl_try,
            tensor_split,
            split_mode,
            ctx_size,
            n_cpu_moe,
            cpu_moe,
            mmproj=mmproj,
            jinja=jinja,
            reasoning_format=reasoning_format,
            no_context_shift=no_context_shift,
        )
        if _wait_for_listen(host, port, timeout=90.0):
            return proc
        # process did not start listening; back off
        terminate_process(proc)
        unregister_process(proc)
        if ngl_try <= 0:
            die("llama-server failed to start even with CPU-only; giving up")
        ngl_try = max(0, ngl_try // 2)
        info(f"[llama][retry] lowering --n-gpu-layers to {ngl_try} (spill remainder to CPU)")

# --------------------------- Transformers Reranker (Qwen) ---------------------------

class TRHandler(BaseHTTPRequestHandler):
    tok = None
    mdl = None
    # If device_map='auto', we keep inputs on CPU and let Accelerate handle device placement.
    input_target_device: Optional[str] = None
    on_cuda: bool = False

    max_len = 4096
    doc_batch = 64
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    yes_id = None
    no_id = None

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "mode": "transformers-rerank"})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": "qwen-reranker", "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    @classmethod
    def _format_pair(cls, instruction: str, query: str, doc: str) -> str:
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        content = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return prefix + content + suffix

    @classmethod
    def _build_inputs(cls, pairs: List[str]):
        tok = cls.tok
        # Dynamic padding reduces VRAM pressure
        inputs = tok(pairs, padding="longest", truncation=True, max_length=cls.max_len, return_tensors="pt")
        return inputs

    @classmethod
    def _score_pairs(cls, pairs: List[str]) -> List[float]:
        import torch
        scores: List[float] = []
        target_bs = max(1, int(cls.doc_batch))
        i = 0
        while i < len(pairs):
            bs = min(target_bs, len(pairs) - i)
            while True:
                chunk = pairs[i:i+bs]
                inputs = cls._build_inputs(chunk)
                # Only move inputs if we are on a single device setup
                if cls.input_target_device is not None:
                    for k in inputs:
                        if hasattr(inputs[k], "to"):
                            inputs[k] = inputs[k].to(cls.input_target_device)
                try:
                    with torch.no_grad():
                        out = cls.mdl(**inputs)
                    logits = out.logits[:, -1, :]  # [B, V]
                    yes = logits[:, cls.yes_id]
                    no  = logits[:, cls.no_id]
                    two = torch.stack([no, yes], dim=1)
                    prob_yes = torch.softmax(two, dim=1)[:, 1].float().cpu().tolist()
                    scores.extend([float(x) for x in prob_yes])
                    i += bs
                    break
                except torch.cuda.OutOfMemoryError:
                    if cls.on_cuda:
                        torch.cuda.empty_cache()
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                except RuntimeError as e:
                    # Some backends raise generic RuntimeError for OOM
                    if "out of memory" in str(e).lower():
                        if cls.on_cuda:
                            torch.cuda.empty_cache()
                        if bs == 1:
                            raise
                        bs = max(1, bs // 2)
                    else:
                        raise
        return scores

    @staticmethod
    def _maybe_parse_pair_line(line: str) -> Optional[Tuple[str, str]]:
        m = re.search(r"query\s*[:=]\s*(.+?)\s*[\r\n]+\s*document\s*[:=]\s*(.+)\s*$", line, re.IGNORECASE | re.DOTALL)
        if not m: return None
        return (m.group(1).strip(), m.group(2).strip())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send(400, {"error": "invalid json"}); return

        if self.path in ("/v1/embeddings", "/embeddings"):
            arr = req.get("input") or []
            if not isinstance(arr, list) or not arr:
                self._send(400, {"error": "expected 'input': list[str]"}); return

            pairs: List[str] = []
            for line in arr:
                if not isinstance(line, str):
                    self._send(400, {"error": "inputs must be strings"}); return
                parsed = self._maybe_parse_pair_line(line)
                if parsed:
                    q, d = parsed
                    pairs.append(self._format_pair(self.__class__.instruction, q, d))
                else:
                    pairs.append(line)

            scores = self._score_pairs(pairs)
            data = [{"embedding": [float(s)]} for s in scores]
            self._send(200, {"data": data})
            return

        if self.path in ("/rerank", "/v1/rerank"):
            query = req.get("query", "")
            docs_in = req.get("documents", [])
            ret = bool(req.get("return_documents", False))
            instr = req.get("instruction", self.__class__.instruction)
            # accept both top_k and top_n
            top_k = req.get("top_k") or req.get("top_n")

            if not isinstance(query, str) or not isinstance(docs_in, list) or not docs_in:
                self._send(400, {"error": "expected fields: query:str, documents:list"})
                return

            # Accept strings or dicts; extract text from dicts if present.
            def get_text(d):
                if isinstance(d, dict):
                    for k in ("text", "content", "value", "document", "doc"):
                        if k in d and isinstance(d[k], str):
                            return d[k]
                    # fallback: stringify whole dict
                    return json.dumps(d, ensure_ascii=False)
                return str(d)

            docs_text = [get_text(d) for d in docs_in]
            pairs = [self._format_pair(instr, query, t) for t in docs_text]
            scores = self._score_pairs(pairs)

            order = sorted(range(len(docs_in)), key=lambda i: scores[i], reverse=True)
            if isinstance(top_k, int) and top_k > 0:
                order = order[:top_k]

            results = []
            for i in order:
                s = float(scores[i])
                # Jina/Cohere style: document must be an object
                doc_obj = docs_in[i] if isinstance(docs_in[i], dict) else {"text": docs_text[i]}
                item = {
                    "index": i,
                    "relevance_score": s,
                }
                if ret:
                    item["document"] = doc_obj

                # Back-compat with existing tooling
                item["score"] = s
                if isinstance(docs_in[i], str):
                    item["document_text"] = docs_text[i]

                results.append(item)

            self._send(200, {"results": results})
            return

        self._send(404, {"error": "not found"})

def serve_transformers_rerank(
    model_ref: str,
    host: str,
    port: int,
    device: str,
    device_map: str,
    dtype: str,
    quant: str,
    doc_batch: int,
    max_len: int,
    max_memory: Optional[str],
    trust_remote_code: bool,
    instruction: Optional[str],
):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        die("Need transformers+torch installed. " + str(e))

    # dtype
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        die(f"Unsupported --dtype: {dtype}")

    model_kwargs = dict(
        trust_remote_code=bool(trust_remote_code),
    )

    # Quantization (bnb) with graceful fallback
    use_bnb = quant.lower() in ("8bit", "4bit")
    if use_bnb:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            _ = bnb.nn.Linear8bitLt  # force-sanity check
            mm = parse_max_memory_arg(max_memory)
            cpu_spill = bool(mm and any((isinstance(k, str) and k == "cpu") for k in mm.keys()))
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(quant.lower() == "8bit"),
                load_in_4bit=(quant.lower() == "4bit"),
                llm_int8_enable_fp32_cpu_offload=(cpu_spill if quant.lower() == "8bit" else False),
            )
            if mm:
                model_kwargs["max_memory"] = mm  # allow Accelerate to place some weights on CPU
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch_dtype
        except Exception as e:
            info(
                "[rerank][warn] bitsandbytes requested but not usable: {}\n"
                "[rerank][warn] Falling back to non-quantized load.".format(e)
            )
            use_bnb = False

    if not use_bnb:
        if device_map == "auto":
            model_kwargs["device_map"] = "auto"
            mm = parse_max_memory_arg(max_memory)
            if mm:
                model_kwargs["max_memory"] = mm  # supports CPU spill in non-bnb mode as well
        elif device_map in ("cuda", "cpu"):
            model_kwargs["device_map"] = None
        else:
            die(f"Unsupported --device-map: {device_map}")
        model_kwargs["torch_dtype"] = torch_dtype

    info(f"[rerank] loading: {model_ref} (device={device}, dtype={torch_dtype}, quant={quant}, device_map={model_kwargs.get('device_map')})")
    tok = AutoTokenizer.from_pretrained(model_ref, padding_side="left", trust_remote_code=bool(trust_remote_code))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs).eval()

    device_map_used = model_kwargs.get("device_map")
    input_target_device: Optional[str]
    if device_map_used in (None, False):
        mdl = mdl.to(device)
        input_target_device = device
    else:
        # Let Accelerate handle device placement. Keep inputs on CPU.
        input_target_device = None

    # Prepare handler statics
    import torch as _torch
    TRHandler.tok = tok
    TRHandler.mdl = mdl
    TRHandler.input_target_device = input_target_device
    TRHandler.on_cuda = _torch.cuda.is_available()
    TRHandler.doc_batch = int(doc_batch)
    TRHandler.max_len = int(max_len)
    TRHandler.instruction = instruction or TRHandler.instruction
    TRHandler.yes_id = tok.convert_tokens_to_ids("yes")
    TRHandler.no_id  = tok.convert_tokens_to_ids("no")

    httpd = ThreadingHTTPServer((host, port), TRHandler)
    info(f"[rerank] listening on http://{host}:{port} (POST /v1/embeddings, /rerank)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

# --------------------------- CLI ---------------------------

def main():
    p = argparse.ArgumentParser("loadmodel.py")
    gmode = p.add_mutually_exclusive_group(required=True)
    gmode.add_argument("--llm", action="store_true", help="Run llama-server for text generation")
    gmode.add_argument("--embed", action="store_true", help="Run llama-server for embeddings")
    gmode.add_argument("--rerank", action="store_true", help="Run Transformers-based reranker")

    p.add_argument("model", help="""
For GGUF modes (--llm/--embed):
  - Local .gguf path, or
  - 'org/repo:quant'  (e.g. Qwen/Qwen3-Embedding-8B-GGUF:Q8_0)
  - 'org/repo:file.gguf' (full filename)

For --rerank (Transformers):
  - HF model id, e.g. Qwen/Qwen3-Reranker-8B
  - Or a local directory with config/tokenizer/weights
""")

    # Common
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=45540)
    p.add_argument("--path", default=str(MODELS_DIR), help="GGUF download dir (default: ./models)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (if needed)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra llama-server flags after --extra ...")

    # llama-server runtime
    p.add_argument("--n-gpu-layers", type=int, default=999)
    p.add_argument("--tensor-split", default=None, help='e.g. "50,50" for 2 GPUs (use "auto" for VRAM split)')
    p.add_argument("--split-mode", default=None, help="none|layer|row (GPU split mode)")
    p.add_argument("--ctx-size", type=int, default=None)
    p.add_argument("--n-cpu-moe", type=int, default=None, help="Offload experts for the first N layers to CPU (Mixture-of-Experts models)")
    p.add_argument("--cpu-moe", action="store_true", help="Offload all experts to CPU (Mixture-of-Experts models)")
    p.add_argument("--mmproj", default=None, help="Path to multimodal projector GGUF for vision/image input (e.g. /path/to/mmproj-F16.gguf)")
    p.add_argument("--jinja", action="store_true", help="Enable Jinja chat template — required for per-request thinking toggle via chat_template_kwargs")
    p.add_argument("--reasoning-format", default=None, help="deepseek|none — how llama-server exposes <think> content (deepseek splits into reasoning_content field)")
    p.add_argument("--no-context-shift", action="store_true", help="Disable KV context shift; recommended for thinking models to avoid cutting off mid-reasoning")

    # Transformers reranker runtime
    p.add_argument("--device", default=None, help="cuda|cpu (single device only if device_map!=auto)")
    p.add_argument("--device-map", default="auto", help="auto|cuda|cpu")
    p.add_argument("--dtype", default="bf16", help="auto|bf16|fp16|fp32")
    p.add_argument("--quant", default="8bit", help="none|8bit|4bit")
    p.add_argument("--doc-batch", type=int, default=64)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--max-memory", default=None, help='Examples: "22GiB,22GiB,cpu=128GiB" or "0=22GiB,1=22GiB,cpu=128GiB"')
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--instruction", default=None, help="Custom rerank instruction (default: web search)")

    args = p.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rerank:
        serve_transformers_rerank(
            model_ref=args.model,
            host=args.host,
            port=args.port,
            device=args.device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"),
            device_map=args.device_map,
            dtype=args.dtype,
            quant=args.quant,
            doc_batch=args.doc_batch,
            max_len=args.max_len,
            max_memory=args.max_memory,
            trust_remote_code=args.trust_remote_code,
            instruction=args.instruction,
        )
        return

    # GGUF path resolve
    gguf_path = resolve_gguf(args.model, Path(args.path), args.hf_token)

    # llama-server (LLM or Embeddings)
    extra = args.extra or []
    if args.embed:
        extra = ["--embeddings"] + extra
    if "--flash-attn" not in extra and "-fa" not in extra:
        extra = ["--flash-attn", "on"] + extra

    tensor_split = args.tensor_split
    if tensor_split:
        kind, _ratios, err = memory_utils.parse_tensor_split(tensor_split)
        if kind == "invalid":
            die(err or "Invalid --tensor-split value.")
        if kind == "auto":
            tensor_split = memory_utils.auto_tensor_split()

    split_mode = args.split_mode
    if split_mode and str(split_mode).strip().lower() == "default":
        split_mode = None

    proc = launch_llama_server_with_backoff(
        model_path=gguf_path,
        host=args.host,
        port=args.port,
        extra=extra,
        n_gpu_layers=args.n_gpu_layers,
        tensor_split=tensor_split,
        split_mode=split_mode,
        ctx_size=args.ctx_size,
        n_cpu_moe=args.n_cpu_moe,
        cpu_moe=args.cpu_moe,
        mmproj=args.mmproj,
        jinja=args.jinja,
        reasoning_format=args.reasoning_format,
        no_context_shift=args.no_context_shift,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    finally:
        unregister_process(proc)

if __name__ == "__main__":
    main()
