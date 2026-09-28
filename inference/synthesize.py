#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


DEFAULT_SDP_RATIO = 0.0
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_SCALE_W = 0.8
DEFAULT_SPEED = 1.0
DEFAULT_STYLE_ID = 0
DEFAULT_SEED = 52001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(root: Path) -> dict:
    p = root / "CHECKPOINTS.json"
    if not p.is_file():
        raise FileNotFoundError(f"CHECKPOINTS.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_checkpoint(root: Path, manifest: dict, stage: str, epoch: int) -> tuple[Path, dict]:
    checkpoints = manifest["stages"][stage]["checkpoints"]
    matches = [x for x in checkpoints if int(x["epoch"]) == epoch]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint for stage={stage} epoch={epoch}, "
            f"found {len(matches)}"
        )
    entry = matches[0]
    path = root / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    actual = sha256(path)
    expected = entry["sha256"]
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch:\n"
            f"  file:     {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
    return path, entry


def load_config(root: Path, manifest: dict, stage: str) -> tuple[Path, dict]:
    rel = manifest["stages"][stage]["config"]
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def resolve_speaker_id(stage: str, config: dict, style: str | None, sid: int | None) -> tuple[int, str]:
    spk2id = dict(config["data"]["spk2id"])

    if sid is not None:
        if sid not in set(int(v) for v in spk2id.values()):
            raise ValueError(f"SID {sid} is not present in config spk2id={spk2id}")
        name = next((k for k, v in spk2id.items() if int(v) == sid), f"SID{sid}")
        return sid, name

    if stage == "yocpt":
        if len(spk2id) != 1:
            raise RuntimeError(
                "YO-CPT config unexpectedly contains multiple speakers. "
                "Use --sid explicitly.\n"
                f"spk2id={spk2id}"
            )
        name, value = next(iter(spk2id.items()))
        return int(value), name

    # Dialogs: map human-readable style to the actual frozen config names.
    wanted = style or "neutral"
    wanted_lower = wanted.lower()

    aliases = {
        "neutral": ("neutral",),
        "positive": ("positive", "happy"),
    }
    tokens = aliases[wanted_lower]

    candidates = [
        (name, int(value))
        for name, value in spk2id.items()
        if any(token in name.lower() for token in tokens)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely map style={wanted!r} to Dialogs speaker/style ID.\n"
            f"spk2id={spk2id}\n"
            "Use --sid 0 or --sid 1 explicitly if needed."
        )
    name, value = candidates[0]
    return value, name


def configure_runtime(melo_root: Path, espeak_build: Path | None) -> None:
    if not melo_root.is_dir():
        raise FileNotFoundError(f"MeloTTS root not found: {melo_root}")

    if espeak_build is not None:
        espeak_build = espeak_build.expanduser().resolve()
        binary_dir = espeak_build / "src"
        if not binary_dir.is_dir():
            raise FileNotFoundError(
                f"Expected pinned eSpeak build src directory: {binary_dir}"
            )
        os.environ["PATH"] = f"{binary_dir}:{os.environ.get('PATH', '')}"
        os.environ["ESPEAK_DATA_PATH"] = str(espeak_build)

    sys.path.insert(0, str(melo_root))


def validate_wav(path: Path, expected_sr: int) -> dict:
    info = sf.info(str(path))
    if info.frames <= 0:
        raise RuntimeError(f"Generated WAV is empty: {path}")
    if info.samplerate != expected_sr:
        raise RuntimeError(
            f"Sampling-rate mismatch: expected {expected_sr}, got {info.samplerate}"
        )
    return {
        "samplerate": info.samplerate,
        "frames": info.frames,
        "duration_sec": info.frames / info.samplerate,
        "channels": info.channels,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Synthesize one WAV from a defense YO-CPT or Dialogs checkpoint."
    )
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--melo-root", type=Path, required=True)
    ap.add_argument(
        "--espeak-build",
        type=Path,
        default=None,
        help=(
            "Pinned eSpeak NG build root. For exact project reproduction use "
            ".../tools/espeak-ng-ru-stress-ab2eab9/build"
        ),
    )
    ap.add_argument("--stage", choices=["yocpt", "dialogs"], required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--style",
        choices=["neutral", "positive"],
        default=None,
        help="Dialogs only. Defaults to neutral.",
    )
    ap.add_argument(
        "--sid",
        type=int,
        default=None,
        help="Explicit speaker/style ID override.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sdp-ratio", type=float, default=DEFAULT_SDP_RATIO)
    ap.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    ap.add_argument("--noise-scale-w", type=float, default=DEFAULT_NOISE_SCALE_W)
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    ap.add_argument("--style-id", type=int, default=DEFAULT_STYLE_ID)
    args = ap.parse_args()

    root = args.checkpoint_root.expanduser().resolve()
    melo_root = args.melo_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    manifest = load_manifest(root)
    ckpt, ckpt_entry = resolve_checkpoint(root, manifest, args.stage, args.epoch)
    config_path, config = load_config(root, manifest, args.stage)

    sid, speaker_name = resolve_speaker_id(
        args.stage, config, args.style, args.sid
    )

    configure_runtime(melo_root, args.espeak_build)

    from melo.api import TTS

    seed_all(args.seed)

    tts = TTS(
        language="RU",
        device=args.device,
        use_hf=False,
        config_path=str(config_path),
        ckpt_path=str(ckpt),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    tts.tts_to_file(
        args.text,
        sid,
        output_path=str(output),
        sdp_ratio=args.sdp_ratio,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        speed=args.speed,
        quiet=True,
        style_id=args.style_id,
    )

    sr = int(config["data"]["sampling_rate"])
    wav = validate_wav(output, sr)

    result = {
        "stage": args.stage,
        "epoch": args.epoch,
        "step": ckpt_entry["step"],
        "selected": bool(ckpt_entry["selected"]),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_entry["sha256"],
        "config": str(config_path),
        "speaker_id": sid,
        "speaker_name": speaker_name,
        "style": args.style,
        "seed": args.seed,
        "inference": {
            "sdp_ratio": args.sdp_ratio,
            "noise_scale": args.noise_scale,
            "noise_scale_w": args.noise_scale_w,
            "speed": args.speed,
            "style_id": args.style_id,
            "device": args.device,
        },
        "output": str(output),
        "wav": wav,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
