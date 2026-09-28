#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

# Same deterministic inference domain as eval_v1.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

PROJECT = "${PROJECT_ROOT}"
MELO = "${MELOTTS_ROOT}"

ESPEAK_BUILD = (
    PROJECT
    + "/tools/espeak-ng-ru-stress-ab2eab9/build"
)

os.environ["PATH"] = (
    ESPEAK_BUILD
    + "/src:"
    + os.environ.get("PATH", "")
)

os.environ["ESPEAK_DATA_PATH"] = ESPEAK_BUILD


import gc
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

if torch.cuda.is_available():
    raise RuntimeError(
        "CPU-only inference guard failed"
    )


sys.path.insert(
    0,
    MELO,
)

from melo.api import TTS


P = Path(PROJECT)

OLD = (
    P
    / "artifacts/eval/"
      "tts_trajectory_eval_v1"
)

OUT = (
    P
    / "artifacts/eval/"
      "tts_trajectory_eval_v2"
)

CHECKPOINT_CONTRACT = (
    P
    / "artifacts/frozen/"
      "tts_trajectory_expanded_checkpoint_contract_v2/"
      "contract.json"
)

PROMPT_CONTRACT = (
    P
    / "artifacts/frozen/"
      "defense_qc_prompts_v2/"
      "prompts.json"
)


EXPECTED_CHECKPOINT_CONTRACT_SHA = (
    "c46b5433f5ea566961d3a25aff3f3b3b"
    "7127efed8badbc1df82b112d5a647d91"
)

EXPECTED_PROMPT_SHA = (
    "95f01b936bd30d120fced130cf2d2b10"
    "d5a1c8777cf36d4295d17e2a7a9863aa"
)


INFERENCE = {
    "sdp_ratio": 0,
    "noise_scale": 0.667,
    "noise_scale_w": 0.8,
    "speed": 1.0,
    "style_id": 0,
    "holdout_seed_base": 52000,
    "defense_seed_base": 52000,
    "device": "cpu",
    "torch_threads": 2,
}


DEFENSE_EPOCHS = {
    1,
    10,
    20,
    30,
    38,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def validate_wav(
    path: Path,
    expected_sr: int,
):
    if not path.is_file():
        return False

    try:
        info = sf.info(
            str(path)
        )
    except Exception:
        return False

    return (
        info.frames > 0
        and info.samplerate == expected_sr
    )


def hardlink_or_copy(
    src: Path,
    dst: Path,
):
    if dst.exists():
        return

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        os.link(
            src,
            dst,
        )
    except OSError:
        shutil.copy2(
            src,
            dst,
        )


def checkpoint_output_dir(
    base: Path,
    epoch: int,
    step: int,
):
    return (
        base
        / (
            f"epoch_{epoch:04d}"
            f"_step_{step:07d}"
        )
    )


# ============================================================
# Contracts
# ============================================================

if sha256(
    CHECKPOINT_CONTRACT
) != EXPECTED_CHECKPOINT_CONTRACT_SHA:
    raise RuntimeError(
        "checkpoint contract SHA mismatch"
    )

if sha256(
    PROMPT_CONTRACT
) != EXPECTED_PROMPT_SHA:
    raise RuntimeError(
        "prompt contract SHA mismatch"
    )


contract = json.loads(
    CHECKPOINT_CONTRACT.read_text(
        encoding="utf-8"
    )
)

prompt_contract = json.loads(
    PROMPT_CONTRACT.read_text(
        encoding="utf-8"
    )
)

prompts = prompt_contract[
    "prompts"
]

if len(prompts) != 9:
    raise RuntimeError(
        "defense prompt count != 9"
    )


# ============================================================
# Old v1 inventory
# ============================================================

old_manifest_path = (
    OLD
    / "render_manifest.jsonl"
)

if not old_manifest_path.is_file():
    raise RuntimeError(
        f"missing v1 manifest: "
        f"{old_manifest_path}"
    )

old_manifest = load_jsonl(
    old_manifest_path
)

if len(old_manifest) != 747:
    raise RuntimeError(
        f"v1 manifest rows={len(old_manifest)}"
    )


old_by_key = {}

for r in old_manifest:

    key = (
        r["experiment"],
        int(r["epoch"]),
        r["row_id"],
    )

    if key in old_by_key:
        raise RuntimeError(
            f"duplicate v1 key: {key}"
        )

    old_by_key[key] = r


# ============================================================
# Create v2 root
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

objective_root = (
    OUT
    / "objective"
)

defense_root = (
    OUT
    / "defense_qc"
)

objective_root.mkdir(
    parents=True,
    exist_ok=True,
)

defense_root.mkdir(
    parents=True,
    exist_ok=True,
)


run_contract = {
    "version": 2,
    "checkpoint_contract": {
        "path": str(
            CHECKPOINT_CONTRACT
        ),
        "sha256": (
            EXPECTED_CHECKPOINT_CONTRACT_SHA
        ),
    },
    "prompt_contract": {
        "path": str(
            PROMPT_CONTRACT
        ),
        "sha256": (
            EXPECTED_PROMPT_SHA
        ),
    },
    "source_eval_v1": {
        "path": str(OLD),
        "manifest": str(
            old_manifest_path
        ),
        "manifest_sha256": sha256(
            old_manifest_path
        ),
    },
    "inference": INFERENCE,
    "policy": {
        "existing_v1_audio": (
            "hardlink if possible, else copy"
        ),
        "existing_artifacts_modified": False,
        "post_winner_in_main_plots": False,
        "defense_post_winner_rendered": False,
    },
}


(
    OUT
    / "run_contract.json"
).write_text(
    json.dumps(
        run_contract,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# Prepare objective references + holdout manifests
# ============================================================

holdouts = {}


for experiment in (
    "dialogs",
    "yocpt",
):

    old_holdout_path = (
        OLD
        / experiment
        / "holdout.jsonl"
    )

    rows = load_jsonl(
        old_holdout_path
    )

    if not rows:
        raise RuntimeError(
            f"empty old holdout: "
            f"{old_holdout_path}"
        )

    exp_root = (
        objective_root
        / experiment
    )

    ref_root = (
        exp_root
        / "reference"
    )

    ref_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_holdout_path = (
        exp_root
        / "holdout.jsonl"
    )

    new_rows = []

    for row in rows:

        row_id = row[
            "row_id"
        ]

        src = Path(
            row[
                "reference_local"
            ]
        )

        dst = (
            ref_root
            / f"{row_id}.wav"
        )

        if not src.is_file():
            raise RuntimeError(
                f"missing old reference: {src}"
            )

        hardlink_or_copy(
            src,
            dst,
        )

        info = sf.info(
            str(dst)
        )

        new = dict(row)

        new[
            "reference_local"
        ] = str(dst)

        new[
            "reference_sha256"
        ] = sha256(dst)

        new[
            "reference_seconds"
        ] = (
            info.frames
            / info.samplerate
        )

        new_rows.append(
            new
        )

    with new_holdout_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in new_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    holdouts[
        experiment
    ] = new_rows


# ============================================================
# Counts / contracts
# ============================================================

dialogs_cp = contract[
    "experiments"
]["dialogs"]["checkpoints"]

yocpt_cp = contract[
    "experiments"
]["yocpt"]["checkpoints"]


expected_objective = (
    len(
        holdouts["dialogs"]
    )
    * len(dialogs_cp)
    +
    len(
        holdouts["yocpt"]
    )
    * len(yocpt_cp)
)

if expected_objective != 1300:
    raise RuntimeError(
        f"objective expected "
        f"{expected_objective} != 1300"
    )


expected_defense = (
    9
    * 2
    * len(
        DEFENSE_EPOCHS
    )
)

if expected_defense != 90:
    raise RuntimeError(
        f"defense expected "
        f"{expected_defense} != 90"
    )


print("=" * 78)
print("TTS TRAJECTORY EVAL V2")
print("=" * 78)

print(
    "objective_total:",
    expected_objective,
)

print(
    "defense_total:",
    expected_defense,
)


# ============================================================
# Render helpers
# ============================================================

objective_records = []
defense_records = []

objective_reused = 0
objective_rendered = 0
objective_skipped = 0

defense_rendered = 0
defense_skipped = 0

t0 = time.time()


def get_config(
    experiment,
):
    return Path(
        contract[
            "experiments"
        ][experiment][
            "config"
        ]
    )


def get_sr_spk(
    experiment,
):
    config = json.loads(
        get_config(
            experiment
        ).read_text(
            encoding="utf-8"
        )
    )

    sr = int(
        config[
            "data"
        ][
            "sampling_rate"
        ]
    )

    spk2id = dict(
        config[
            "data"
        ][
            "spk2id"
        ]
    )

    return (
        sr,
        spk2id,
    )


def objective_target(
    experiment,
    epoch,
    step,
    row_id,
):
    return (
        checkpoint_output_dir(
            objective_root
            / experiment
            / "synth",
            epoch,
            step,
        )
        / f"{row_id}.wav"
    )


def defense_target(
    epoch,
    step,
    prompt_id,
    sid,
):
    return (
        checkpoint_output_dir(
            defense_root
            / "dialogs"
            / "synth",
            epoch,
            step,
        )
        / (
            f"{prompt_id}"
            f"_SID{sid}.wav"
        )
    )


# ============================================================
# Process one experiment checkpoint at a time
# ============================================================

for experiment in (
    "dialogs",
    "yocpt",
):

    sr, spk2id = (
        get_sr_spk(
            experiment
        )
    )

    cp_list = contract[
        "experiments"
    ][experiment][
        "checkpoints"
    ]

    holdout = holdouts[
        experiment
    ]

    for cp in cp_list:

        epoch = int(
            cp["epoch"]
        )

        step = int(
            cp["step"]
        )

        checkpoint_role = cp[
            "role"
        ]

        generator = Path(
            cp["G"]
        )

        actual_g_sha = sha256(
            generator
        )

        if (
            actual_g_sha
            != cp["G_sha256"]
        ):
            raise RuntimeError(
                f"{experiment} E{epoch}: "
                "G SHA mismatch"
            )

        #
        # First reuse all objective WAVs already
        # available from eval_v1.
        #
        objective_need_render = []

        for row in holdout:

            row_id = row[
                "row_id"
            ]

            dst = objective_target(
                experiment,
                epoch,
                step,
                row_id,
            )

            if validate_wav(
                dst,
                sr,
            ):
                objective_skipped += 1
                continue

            old_key = (
                experiment,
                epoch,
                row_id,
            )

            old = old_by_key.get(
                old_key
            )

            if old is not None:

                src = Path(
                    old["wav"]
                )

                if not validate_wav(
                    src,
                    sr,
                ):
                    raise RuntimeError(
                        f"invalid v1 source: "
                        f"{src}"
                    )

                actual_old_sha = (
                    sha256(src)
                )

                if (
                    actual_old_sha
                    != old[
                        "wav_sha256"
                    ]
                ):
                    raise RuntimeError(
                        f"v1 WAV SHA mismatch: "
                        f"{src}"
                    )

                if dst.exists():
                    dst.unlink()

                hardlink_or_copy(
                    src,
                    dst,
                )

                if not validate_wav(
                    dst,
                    sr,
                ):
                    raise RuntimeError(
                        f"reuse failed: {dst}"
                    )

                objective_reused += 1

            else:
                objective_need_render.append(
                    row
                )

        #
        # Defense render applies only to dialogs
        # main/winner epochs, not E40.
        #
        defense_need_render = []

        if (
            experiment == "dialogs"
            and epoch in DEFENSE_EPOCHS
        ):
            for prompt in prompts:
                prompt_id = prompt["id"]

                for sid in (
                    0,
                    1,
                ):

                    dst = defense_target(
                        epoch,
                        step,
                        prompt_id,
                        sid,
                    )

                    if validate_wav(
                        dst,
                        sr,
                    ):
                        defense_skipped += 1
                    else:
                        defense_need_render.append(
                            (
                                prompt,
                                sid,
                            )
                        )

        #
        # Nothing new for this checkpoint.
        #
        if (
            not objective_need_render
            and not defense_need_render
        ):
            print(
                f"{experiment.upper()} "
                f"E{epoch:02d}: "
                "all required audio present",
                flush=True,
            )

            continue

        print()
        print("=" * 78)
        print(
            f"LOAD {experiment.upper()} "
            f"E{epoch:02d} "
            f"step={step} "
            f"role={checkpoint_role}"
        )
        print(
            "objective_to_render:",
            len(
                objective_need_render
            ),
        )
        print(
            "defense_to_render:",
            len(
                defense_need_render
            ),
        )
        print("=" * 78)

        seed_all(
            52000
        )

        tts = TTS(
            language="RU",
            device="cpu",
            use_hf=False,
            config_path=str(
                get_config(
                    experiment
                )
            ),
            ckpt_path=str(
                generator
            ),
        )

        #
        # Objective holdout.
        #
        for row in objective_need_render:

            row_id = row[
                "row_id"
            ]

            seed = int(
                row["seed"]
            )

            if experiment == "dialogs":
                speaker_name = row[
                    "speaker"
                ]

                sid = int(
                    spk2id[
                        speaker_name
                    ]
                )

            else:
                sid = int(
                    next(
                        iter(
                            spk2id.values()
                        )
                    )
                )

            dst = objective_target(
                experiment,
                epoch,
                step,
                row_id,
            )

            dst.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if dst.exists():
                dst.unlink()

            seed_all(seed)

            tts.tts_to_file(
                row["text"],
                sid,
                output_path=str(dst),
                sdp_ratio=(
                    INFERENCE[
                        "sdp_ratio"
                    ]
                ),
                noise_scale=(
                    INFERENCE[
                        "noise_scale"
                    ]
                ),
                noise_scale_w=(
                    INFERENCE[
                        "noise_scale_w"
                    ]
                ),
                speed=(
                    INFERENCE[
                        "speed"
                    ]
                ),
                quiet=True,
                style_id=(
                    INFERENCE[
                        "style_id"
                    ]
                ),
            )

            if not validate_wav(
                dst,
                sr,
            ):
                raise RuntimeError(
                    f"invalid objective render: "
                    f"{dst}"
                )

            objective_rendered += 1

            print(
                f"OBJECTIVE "
                f"[{objective_rendered:03d}/747] "
                f"{experiment} "
                f"E{epoch:02d} "
                f"{row_id}",
                flush=True,
            )

        #
        # Controlled defense prompts.
        #
        for prompt, sid in (
            defense_need_render
        ):

            prompt_num = int(
                prompt["id"][1:]
            )

            seed = (
                INFERENCE[
                    "defense_seed_base"
                ]
                + prompt_num
            )

            dst = defense_target(
                epoch,
                step,
                prompt["id"],
                sid,
            )

            dst.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if dst.exists():
                dst.unlink()

            #
            # Crucial paired contract:
            # SID0 and SID1 get the SAME seed.
            #
            seed_all(seed)

            tts.tts_to_file(
                prompt["text"],
                sid,
                output_path=str(dst),
                sdp_ratio=(
                    INFERENCE[
                        "sdp_ratio"
                    ]
                ),
                noise_scale=(
                    INFERENCE[
                        "noise_scale"
                    ]
                ),
                noise_scale_w=(
                    INFERENCE[
                        "noise_scale_w"
                    ]
                ),
                speed=(
                    INFERENCE[
                        "speed"
                    ]
                ),
                quiet=True,
                style_id=(
                    INFERENCE[
                        "style_id"
                    ]
                ),
            )

            if not validate_wav(
                dst,
                sr,
            ):
                raise RuntimeError(
                    f"invalid defense render: "
                    f"{dst}"
                )

            defense_rendered += 1

            print(
                f"DEFENSE "
                f"[{defense_rendered:02d}/90] "
                f"E{epoch:02d} "
                f"{prompt['id']} "
                f"SID{sid}",
                flush=True,
            )

        del tts
        gc.collect()


# ============================================================
# Final objective manifest
# ============================================================

for experiment in (
    "dialogs",
    "yocpt",
):

    sr, spk2id = (
        get_sr_spk(
            experiment
        )
    )

    cp_list = contract[
        "experiments"
    ][experiment][
        "checkpoints"
    ]

    for cp in cp_list:

        epoch = int(
            cp["epoch"]
        )

        step = int(
            cp["step"]
        )

        for row in holdouts[
            experiment
        ]:

            row_id = row[
                "row_id"
            ]

            wav = objective_target(
                experiment,
                epoch,
                step,
                row_id,
            )

            if not validate_wav(
                wav,
                sr,
            ):
                raise RuntimeError(
                    f"final objective missing: "
                    f"{wav}"
                )

            info = sf.info(
                str(wav)
            )

            if experiment == "dialogs":
                sid = int(
                    spk2id[
                        row[
                            "speaker"
                        ]
                    ]
                )
            else:
                sid = int(
                    next(
                        iter(
                            spk2id.values()
                        )
                    )
                )

            objective_records.append(
                {
                    "experiment": (
                        experiment
                    ),
                    "holdout_kind": (
                        contract[
                            "experiments"
                        ][experiment][
                            "holdout_kind"
                        ]
                    ),
                    "row_id": row_id,
                    "speaker": row[
                        "speaker"
                    ],
                    "speaker_id": sid,
                    "text": row[
                        "text"
                    ],
                    "seed": int(
                        row["seed"]
                    ),
                    "checkpoint_role": (
                        cp["role"]
                    ),
                    "epoch": epoch,
                    "step": step,
                    "generator_sha256": (
                        cp[
                            "G_sha256"
                        ]
                    ),
                    "wav": str(wav),
                    "wav_sha256": (
                        sha256(wav)
                    ),
                    "sampling_rate": (
                        info.samplerate
                    ),
                    "frames": (
                        info.frames
                    ),
                    "duration_sec": (
                        info.frames
                        / info.samplerate
                    ),
                }
            )


if len(
    objective_records
) != 1300:
    raise RuntimeError(
        "objective manifest count "
        f"{len(objective_records)} "
        "!= 1300"
    )


objective_manifest = (
    OUT
    / "objective_manifest.jsonl"
)

with objective_manifest.open(
    "w",
    encoding="utf-8",
) as f:

    for r in objective_records:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Final defense manifest
# ============================================================

dialogs_sr, dialogs_spk2id = (
    get_sr_spk(
        "dialogs"
    )
)

sid2name = {
    int(v): k
    for k, v in (
        dialogs_spk2id.items()
    )
}


dialogs_cp_by_epoch = {
    int(cp["epoch"]): cp
    for cp in dialogs_cp
}


for epoch in sorted(
    DEFENSE_EPOCHS
):

    cp = dialogs_cp_by_epoch[
        epoch
    ]

    step = int(
        cp["step"]
    )

    for prompt in prompts:

        prompt_num = int(
            prompt["id"][1:]
        )

        seed = (
            INFERENCE[
                "defense_seed_base"
            ]
            + prompt_num
        )

        for sid in (
            0,
            1,
        ):

            wav = defense_target(
                epoch,
                step,
                prompt["id"],
                sid,
            )

            if not validate_wav(
                wav,
                dialogs_sr,
            ):
                raise RuntimeError(
                    f"final defense missing: "
                    f"{wav}"
                )

            info = sf.info(
                str(wav)
            )

            defense_records.append(
                {
                    "experiment": "dialogs",
                    "prompt_id": (
                        prompt["id"]
                    ),
                    "category": (
                        prompt[
                            "category"
                        ]
                    ),
                    "text": (
                        prompt["text"]
                    ),
                    "epoch": epoch,
                    "step": step,
                    "checkpoint_role": (
                        cp["role"]
                    ),
                    "speaker_id": sid,
                    "speaker": (
                        sid2name[sid]
                    ),
                    "seed": seed,
                    "generator_sha256": (
                        cp[
                            "G_sha256"
                        ]
                    ),
                    "wav": str(wav),
                    "wav_sha256": (
                        sha256(wav)
                    ),
                    "sampling_rate": (
                        info.samplerate
                    ),
                    "duration_sec": (
                        info.frames
                        / info.samplerate
                    ),
                }
            )


if len(
    defense_records
) != 90:
    raise RuntimeError(
        "defense manifest count "
        f"{len(defense_records)} "
        "!= 90"
    )


defense_manifest = (
    OUT
    / "defense_manifest.jsonl"
)

with defense_manifest.open(
    "w",
    encoding="utf-8",
) as f:

    for r in defense_records:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Checksums
# ============================================================

checksum_file = (
    OUT
    / "SHA256SUMS.audio.txt"
)

with checksum_file.open(
    "w",
    encoding="utf-8",
) as f:

    #
    # References.
    #
    for experiment in (
        "dialogs",
        "yocpt",
    ):
        for row in holdouts[
            experiment
        ]:
            wav = Path(
                row[
                    "reference_local"
                ]
            )

            f.write(
                f"{sha256(wav)}  "
                f"{wav.relative_to(OUT)}\n"
            )

    #
    # Objective synthetic.
    #
    for r in objective_records:
        wav = Path(
            r["wav"]
        )

        f.write(
            f"{r['wav_sha256']}  "
            f"{wav.relative_to(OUT)}\n"
        )

    #
    # Defense synthetic.
    #
    for r in defense_records:
        wav = Path(
            r["wav"]
        )

        f.write(
            f"{r['wav_sha256']}  "
            f"{wav.relative_to(OUT)}\n"
        )


elapsed = (
    time.time()
    - t0
)


summary = {
    "status": "PASS",

    "objective": {
        "total": 1300,
        "reused_from_v1_this_run": (
            objective_reused
        ),
        "rendered_this_run": (
            objective_rendered
        ),
        "already_present_this_run": (
            objective_skipped
        ),
        "dialogs": {
            "epochs": [
                1,
                10,
                20,
                30,
                38,
                40,
            ],
            "post_winner": 40,
        },
        "yocpt": {
            "epochs": [
                1,
                10,
                20,
                27,
                30,
            ],
            "post_winner": 30,
        },
    },

    "defense_qc": {
        "total": 90,
        "rendered_this_run": (
            defense_rendered
        ),
        "already_present_this_run": (
            defense_skipped
        ),
        "epochs": [
            1,
            10,
            20,
            30,
            38,
        ],
        "prompts": 9,
        "speakers": 2,
        "paired_seed": True,
    },

    "elapsed_sec": elapsed,
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


(
    OUT
    / ".complete"
).write_text(
    "PASS\n",
    encoding="utf-8",
)


print()
print("=" * 78)
print("TTS TRAJECTORY EVAL V2: PASS")
print("=" * 78)

print(
    "objective_manifest_rows:",
    len(
        objective_records
    ),
)

print(
    "objective_reused_from_v1_this_run:",
    objective_reused,
)

print(
    "objective_rendered_this_run:",
    objective_rendered,
)

print(
    "objective_already_present_this_run:",
    objective_skipped,
)

print(
    "defense_manifest_rows:",
    len(
        defense_records
    ),
)

print(
    "defense_rendered_this_run:",
    defense_rendered,
)

print(
    "defense_already_present_this_run:",
    defense_skipped,
)

print(
    "elapsed_sec:",
    round(
        elapsed,
        2,
    ),
)

print(
    "output:",
    OUT,
)
