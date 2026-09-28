#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EPOCHS = {
    "yocpt": [1, 10, 20, 27, 30],
    "dialogs": [1, 10, 20, 30, 38, 40],
}
CHOSEN = {
    "yocpt": 27,
    "dialogs": 38,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render the same text across all defense epochs."
    )
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--melo-root", type=Path, required=True)
    ap.add_argument("--espeak-build", type=Path, default=None)
    ap.add_argument("--stage", choices=["yocpt", "dialogs"], required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--style",
        choices=["neutral", "positive"],
        default=None,
        help="Dialogs only. Defaults to neutral.",
    )
    ap.add_argument("--sid", type=int, default=None)
    ap.add_argument("--seed", type=int, default=52001)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    script = Path(__file__).resolve().with_name("synthesize.py")
    if not script.is_file():
        raise FileNotFoundError(f"Missing sibling synthesize.py: {script}")

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    records = []

    for epoch in EPOCHS[args.stage]:
        suffix = "_chosen" if epoch == CHOSEN[args.stage] else ""
        style_suffix = ""
        if args.stage == "dialogs":
            style_suffix = f"_{args.style or 'neutral'}"

        wav = out / f"E{epoch:02d}{suffix}{style_suffix}.wav"

        cmd = [
            sys.executable,
            str(script),
            "--checkpoint-root", str(args.checkpoint_root),
            "--melo-root", str(args.melo_root),
            "--stage", args.stage,
            "--epoch", str(epoch),
            "--text", args.text,
            "--output", str(wav),
            "--seed", str(args.seed),
            "--device", args.device,
        ]

        if args.espeak_build is not None:
            cmd += ["--espeak-build", str(args.espeak_build)]
        if args.style is not None:
            cmd += ["--style", args.style]
        if args.sid is not None:
            cmd += ["--sid", str(args.sid)]

        print("=" * 78, flush=True)
        print(
            f"RENDER {args.stage.upper()} E{epoch:02d}"
            f"{' CHOSEN' if epoch == CHOSEN[args.stage] else ''}",
            flush=True,
        )
        print("=" * 78, flush=True)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"synthesize.py failed for {args.stage} E{epoch:02d} "
                f"with exit code {proc.returncode}"
            )

        result = json.loads(proc.stdout)
        records.append(result)

        print(
            f"OK {wav.name}: "
            f"{result['wav']['duration_sec']:.3f}s, "
            f"SID={result['speaker_id']}",
            flush=True,
        )

    manifest = {
        "stage": args.stage,
        "text": args.text,
        "style": args.style,
        "sid_override": args.sid,
        "seed": args.seed,
        "device": args.device,
        "epochs": records,
    }

    manifest_path = out / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("PASS: trajectory rendered")
    print("output:", out)
    print("manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
