"""Network access to the PhysioNet open S3 mirror.

The bucket is public and unsigned, so plain HTTPS is enough -- no AWS CLI and
no wget on this machine (RULES Appendix B). urllib covers listing, range reads
and whole-object reads; resume is handled by a Range header rather than
`curl -C -` so there is one code path.

Nothing here caches. Callers own the disk lifecycle, because deletion is a
one-way door (R11).
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

BUCKET = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "chbmit/1.0.0/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

RETRIES = 4
BACKOFF = 2.0
TIMEOUT = 120


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    etag: str

    @property
    def rel(self) -> str:
        return self.key[len(PREFIX) :] if self.key.startswith(PREFIX) else self.key


def _open(url: str, headers: dict[str, str] | None = None) -> bytes:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}") from last


def list_objects(prefix: str = PREFIX) -> list[S3Object]:
    """Full recursive listing, following continuation tokens."""
    out: list[S3Object] = []
    token: str | None = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(_open(BUCKET + "?" + urllib.parse.urlencode(q)))
        for c in root.findall(NS + "Contents"):
            out.append(
                S3Object(
                    key=c.findtext(NS + "Key", ""),
                    size=int(c.findtext(NS + "Size", "0")),
                    etag=(c.findtext(NS + "ETag", "") or "").strip('"'),
                )
            )
        if (root.findtext(NS + "IsTruncated") or "false") != "true":
            return out
        token = root.findtext(NS + "NextContinuationToken")


def url_for(rel: str) -> str:
    """Percent-encode the key path.

    ``chb02_16+.edf`` is a real filename. An unencoded ``+`` in a URL path is
    read as a space by S3, so the object 404s. PRD §2 flags the chb17a/chb17b
    naming variant but not this one.
    """
    return BUCKET + PREFIX + urllib.parse.quote(rel.lstrip("/"), safe="/")


def get_range(rel: str, start: int, end: int) -> bytes:
    """Inclusive byte range. R15 exempts header reads from the no-two-passes
    rule, and this is the call that makes R30 cheap."""
    return _open(url_for(rel), {"Range": f"bytes={start}-{end}"})


def get_bytes(rel: str) -> bytes:
    return _open(url_for(rel))


def get_text(rel: str) -> str:
    return get_bytes(rel).decode("utf-8", "replace")


def download(rel: str, dest: Path, expect_size: int | None = None) -> Path:
    """Resumable whole-object download. Appends from wherever a partial file
    left off, so an interrupted run costs only the tail."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expect_size is not None and have == expect_size:
        return dest
    if expect_size is not None and have > expect_size:
        dest.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    chunk = _open(url_for(rel), headers)
    with open(dest, "ab" if have else "wb") as f:
        f.write(chunk)
    if expect_size is not None and dest.stat().st_size != expect_size:
        raise RuntimeError(
            f"{rel}: expected {expect_size} B, got {dest.stat().st_size} B"
        )
    return dest


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    """SHA256SUMS.txt -> {relative_path: digest}.

    R13: there is no MD5SUMS in this distribution, contrary to PRD §2 and M1.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            out[parts[-1].lstrip("*./")] = parts[0].lower()
    return out


def iter_edf_keys(objs: list[S3Object]) -> Iterator[S3Object]:
    for o in objs:
        if o.key.endswith(".edf"):
            yield o
