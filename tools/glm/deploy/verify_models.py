#!/usr/bin/env python3
"""
Check a downloaded GLM-5.3 Ablit pack and DFlash2 drafter against model-manifest.json (sizes always, hashes with --hash).

    python3 verify_models.py --models-dir /srv/models [--hash]

Expects <models-dir>/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit and <models-dir>/GLM-5.3-Flash-DFlash2. LFS files carry a
sha256; small git files carry the git blob sha1 (sha1 of "blob <len>\\0" + content), as on the Hugging Face hub.
Download with: hf download <repo> --revision <revision> --local-dir <models-dir>/<name>
"""
import argparse, hashlib, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--models-dir", required = True)
ap.add_argument("--hash", action = "store_true", help = "also hash every file (reads ~166 GiB)")
args = ap.parse_args()
manifest = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-manifest.json")))
bad = 0
for name, m in manifest.items():
    print(f"{name}: {m['repo']} @ {m['revision']}")
    for rel, f in m["files"].items():
        p = os.path.join(args.models_dir, name, rel)
        if not os.path.exists(p):
            print(f"  MISSING {rel}"); bad += 1; continue
        if os.path.getsize(p) != f["bytes"]:
            print(f"  SIZE    {rel}: {os.path.getsize(p)} != {f['bytes']}"); bad += 1; continue
        if args.hash:
            if "sha256" in f:
                h = hashlib.sha256()
            else:
                h = hashlib.sha1(); h.update(b"blob %d\0" % f["bytes"])
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != f.get("sha256", f.get("git_blob_sha1")):
                print(f"  HASH    {rel}"); bad += 1; continue
    print(f"  {len(m['files'])} files checked")
print("OK" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
