import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA

LOCK = threading.RLock()


def uid():
    return uuid.uuid4().hex


def now():
    return datetime.now(timezone.utc).isoformat()


def path_for(kind: str, ident: str, ext="json") -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", ident):
        raise ValueError("Invalid record identifier")
    return DATA / kind / f"{ident}.{ext}"


def write_json(path: Path, value):
    with LOCK:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
        temp.replace(path)


def records(kind):
    return sorted(
        [json.loads(p.read_text()) for p in (DATA / kind).glob("*.json")],
        key=lambda r: r.get("created", ""),
        reverse=True,
    )


def read(kind, ident):
    return json.loads(path_for(kind, ident).read_text())


def save(kind, value):
    value = {"id": uid(), "created": now(), **value}
    write_json(path_for(kind, value["id"]), value)
    return value
