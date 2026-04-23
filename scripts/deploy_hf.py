#!/usr/bin/env python
"""One-shot HF Spaces deployer.

Usage:
    export HF_TOKEN=<write-scoped token from https://huggingface.co/settings/tokens>
    python scripts/deploy_hf.py

Creates (or updates) the Space at huggingface.co/spaces/<user>/<repo> by
uploading every file under deploy/hf_space/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="lkshay")
    p.add_argument("--space", default="medibot")
    p.add_argument("--sdk", default="gradio", choices=["gradio", "streamlit", "docker"])
    p.add_argument("--dir", default="deploy/hf_space", help="Local dir to upload.")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN env (needs 'write' scope).", file=sys.stderr)
        print("Create one at https://huggingface.co/settings/tokens", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, create_repo

    repo_id = f"{args.user}/{args.space}"
    api = HfApi(token=token)

    print(f"Ensuring Space exists: {repo_id} (sdk={args.sdk}) ...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=args.sdk,
            token=token,
            exist_ok=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"create_repo: {e}")

    src = Path(args.dir)
    if not src.is_dir():
        print(f"ERROR: {src} is not a directory.", file=sys.stderr)
        return 1

    print(f"Uploading files from {src}/ ...")
    uploaded = 0
    for path in sorted(src.rglob("*")):
        if path.is_dir() or path.name.startswith(".") or path.name == "__pycache__":
            continue
        rel = path.relative_to(src).as_posix()
        print(f"  -> {rel}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=repo_id,
            repo_type="space",
        )
        uploaded += 1

    print(f"\nUploaded {uploaded} files.")
    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print(f"Settings:  https://huggingface.co/spaces/{repo_id}/settings")
    print()
    print("Next step — add the runtime secret:")
    print(f"  HUGGINGFACEHUB_API_TOKEN = <any HF token with Inference read access>")
    print("  (set it in Space Settings -> Variables and secrets -> New secret)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
