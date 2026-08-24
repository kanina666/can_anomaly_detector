"""
Fetches the ~945MB of first_pos_v2 external artifacts (4 PatchCore memory
banks + 2 pretrained backbone checkpoints) that are NOT committed to git
(too large for a normal GitHub push -- see system/README.md).

Source: a public Yandex.Disk folder, configured in artifacts_source.json
("yandex_disk_public_url"). Every downloaded file is verified against
artifacts/checksums_external.json (sha256) before being accepted; already-
present files that already match are skipped, so this is safe to re-run.

Usage:
    python download_artifacts.py
    python download_artifacts.py --url https://disk.yandex.ru/d/XXXXXXXX
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "artifacts_external")
CHECKSUMS_PATH = os.path.join(HERE, "artifacts", "checksums_external.json")
SOURCE_CONFIG_PATH = os.path.join(HERE, "artifacts_source.json")

API_BASE = "https://cloud-api.yandex.net/v1/disk/public/resources"


def _sha256(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_valid(name, expected):
    path = os.path.join(DEST, name)
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) != expected["size_bytes"]:
        return False
    return _sha256(path) == expected["sha256"]


def _list_public_folder(public_url):
    req = urllib.request.Request(f"{API_BASE}?public_key={urllib.parse.quote(public_url, safe='')}&limit=100")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    items = data.get("_embedded", {}).get("items", [])
    return {item["name"]: item for item in items if item.get("type") == "file"}


def _download_file(public_url, item_path, name, dest_path):
    q = urllib.parse.urlencode({"public_key": public_url, "path": item_path})
    with urllib.request.urlopen(f"{API_BASE}/download?{q}", timeout=30) as resp:
        href = json.load(resp)["href"]
    tmp_path = dest_path + ".part"
    with urllib.request.urlopen(href, timeout=60) as resp, open(tmp_path, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_path, dest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Public Yandex.Disk folder URL (overrides artifacts_source.json)")
    args = parser.parse_args()

    with open(CHECKSUMS_PATH, encoding="utf-8") as f:
        checksums = json.load(f)

    public_url = args.url
    if not public_url:
        with open(SOURCE_CONFIG_PATH, encoding="utf-8") as f:
            public_url = json.load(f).get("yandex_disk_public_url", "")
    if not public_url:
        print("No source URL configured. Set 'yandex_disk_public_url' in "
              "artifacts_source.json, or pass --url.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DEST, exist_ok=True)

    to_fetch = {name: meta for name, meta in checksums.items() if not _already_valid(name, meta)}
    if not to_fetch:
        print("All external artifacts already present and verified.")
        return

    print(f"Listing {public_url} ...")
    remote_items = _list_public_folder(public_url)

    for name, meta in to_fetch.items():
        if name not in remote_items:
            print(f"  MISSING on remote: {name}", file=sys.stderr)
            continue
        dest_path = os.path.join(DEST, name)
        print(f"  downloading {name} ({meta['size_bytes'] / 1e6:.0f} MB) ...")
        _download_file(public_url, remote_items[name]["path"], name, dest_path)
        if not _already_valid(name, meta):
            os.remove(dest_path)
            print(f"  CHECKSUM MISMATCH after download: {name} -- removed, re-run to retry.", file=sys.stderr)
        else:
            print(f"  ok: {name}")

    missing = [n for n in checksums if not _already_valid(n, checksums[n])]
    if missing:
        print(f"\nStill missing/invalid: {missing}", file=sys.stderr)
        sys.exit(1)
    print("\nAll external artifacts present and verified.")


if __name__ == "__main__":
    main()
