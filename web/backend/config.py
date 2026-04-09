"""Load and persist the web backend's configuration file."""
from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / ".web_config.json"
STATE_PATH = REPO_ROOT / ".web_state.json"
LOG_DIR = REPO_ROOT / "web" / "logs"
FRONTEND_DIST = REPO_ROOT / "web" / "frontend" / "dist"
MODELS_DIR_DEFAULT = REPO_ROOT / "models"


@dataclass
class WebConfig:
    token: str = ""
    host: str = "0.0.0.0"
    port: int = 8787
    models_dir: str = str(MODELS_DIR_DEFAULT)
    cors_origins: list = field(default_factory=lambda: ["*"])

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir).expanduser().resolve()


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def load_config(path: Path = CONFIG_PATH) -> WebConfig:
    """Load config from disk, creating a fresh one with a generated token if missing."""
    if not path.exists():
        cfg = WebConfig(token=_generate_token())
        save_config(cfg, path)
        return cfg
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {path}: {exc}") from exc
    cfg = WebConfig(
        token=str(raw.get("token") or ""),
        host=str(raw.get("host") or "0.0.0.0"),
        port=int(raw.get("port") or 8787),
        models_dir=str(raw.get("models_dir") or MODELS_DIR_DEFAULT),
        cors_origins=list(raw.get("cors_origins") or ["*"]),
    )
    changed = False
    if not cfg.token:
        cfg.token = _generate_token()
        changed = True
    if changed:
        save_config(cfg, path)
    return cfg


def save_config(cfg: WebConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n")


def init_config(path: Path = CONFIG_PATH, *, force: bool = False) -> WebConfig:
    """Write a fresh config (or keep existing) and return it."""
    if path.exists() and not force:
        return load_config(path)
    cfg = WebConfig(token=_generate_token())
    save_config(cfg, path)
    return cfg


def ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR
