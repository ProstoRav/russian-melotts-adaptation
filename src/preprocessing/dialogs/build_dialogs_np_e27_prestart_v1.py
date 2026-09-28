from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch


PROJECT = Path(
    "${PROJECT_ROOT}"
)

FOUNDATION_DIR = (
    PROJECT
    / "artifacts/qc/"
      "spk298_prod_epoch_qc_v1/checkpoints/"
      "epoch_0027_step_0056645"
)

SRC_G = FOUNDATION_DIR / "G_56645.pth"
SRC_D = FOUNDATION_DIR / "D_56645.pth"
SRC_DUR = FOUNDATION_DIR / "DUR_56645.pth"

SRC_CONFIG = (
    PROJECT
    / "runs/"
      "spk298_from_englishv3_prod_b2w6_noeval_v1/"
      "model/config.json"
)

DATASET = (
    PROJECT
    / "artifacts/frozen/"
      "dialogs_np_melotts_no_bert_stress_v1"
)

OUT = (
    PROJECT
    / "artifacts/frozen/"
      "dialogs_np_e27_prestart_v1"
)

TMP = Path(
    str(OUT) + ".tmp"
)


EXPECTED_SHA = {
    "G":
        "7d03e90f9b94ecca28ab62246078ef3431708ba5392d129d7ddde5a2420cf2cc",

    "D":
        "609bf16611957c51c4836d390d88f4b684ba74709a517513424d4b9afbbcab34",

    "DUR":
        "71194eecb88205da06cfecc36da66628caa5210bb050ab215108d5d87e809620",

    "config":
        "9a38d22fa44ab8fd7249bf96fc5871a5bc6aa4fb06c4ef964846d5aba61622ad",
}


SPEAKER_MAP = {
    "dialogs_female_neutral": 0,
    "dialogs_female_positive": 1,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_ckpt(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def fail(message: str):
    raise RuntimeError(
        message
    )


print(
    "===== DIALOGS NP E27 PRESTART V1 =====",
    flush=True,
)


if OUT.exists():
    fail(
        f"refusing to overwrite frozen artifact: {OUT}"
    )

if TMP.exists():
    fail(
        f"stale temp artifact exists: {TMP}"
    )


#
# Source integrity.
#
for label, path in [
    ("G", SRC_G),
    ("D", SRC_D),
    ("DUR", SRC_DUR),
    ("config", SRC_CONFIG),
]:
    if not path.is_file():
        fail(
            f"{label} missing: {path}"
        )

    actual = sha256(
        path
    )

    expected = EXPECTED_SHA[
        label
    ]

    if actual != expected:
        fail(
            f"{label} SHA mismatch: "
            f"expected={expected} "
            f"actual={actual}"
        )

    print(
        f"{label} source SHA256: PASS",
        flush=True,
    )


#
# Frozen dataset integrity.
#
dataset_sums = (
    DATASET / "SHA256SUMS.txt"
)

if not dataset_sums.is_file():
    fail(
        f"dataset SHA256SUMS missing: "
        f"{dataset_sums}"
    )

for filename in [
    "train.list",
    "val.list",
    "test.list",
    "speaker_map.json",
    "summary.json",
    "wav_sha256.tsv",
]:
    if not (
        DATASET / filename
    ).is_file():
        fail(
            f"dataset file missing: {filename}"
        )


with (
    DATASET / "speaker_map.json"
).open(
    "r",
    encoding="utf-8",
) as f:
    dataset_speaker_map = json.load(
        f
    )

if (
    dataset_speaker_map["spk2id"]
    != SPEAKER_MAP
):
    fail(
        "frozen dataset speaker map mismatch"
    )

if (
    int(
        dataset_speaker_map["n_speakers"]
    )
    != 2
):
    fail(
        "frozen dataset n_speakers != 2"
    )


#
# Load source triplet.
#
g = load_ckpt(
    SRC_G
)

d = load_ckpt(
    SRC_D
)

dur = load_ckpt(
    SRC_DUR
)


for label, ckpt in [
    ("G", g),
    ("D", d),
    ("DUR", dur),
]:
    required = {
        "model",
        "optimizer",
        "learning_rate",
        "iteration",
    }

    missing = required - set(
        ckpt.keys()
    )

    if missing:
        fail(
            f"{label} missing keys: "
            f"{sorted(missing)}"
        )

    if ckpt["optimizer"] is None:
        fail(
            f"{label} source optimizer unexpectedly None"
        )

    if int(
        ckpt["iteration"]
    ) != 27:
        fail(
            f"{label} source iteration != 27: "
            f"{ckpt['iteration']}"
        )


#
# Generator 1 -> 2 speaker expansion.
#
state_g = g[
    "model"
]

if "emb_g.weight" not in state_g:
    fail(
        "source G has no emb_g.weight"
    )

old_emb = state_g[
    "emb_g.weight"
]

if tuple(
    old_emb.shape
) != (1, 256):
    fail(
        f"unexpected source emb_g shape: "
        f"{tuple(old_emb.shape)}"
    )


old_row = (
    old_emb[0]
    .detach()
    .cpu()
    .clone()
)


new_emb = torch.stack(
    [
        old_row.clone(),
        old_row.clone(),
    ],
    dim=0,
)


if tuple(
    new_emb.shape
) != (2, 256):
    fail(
        "expanded emb_g shape mismatch"
    )

if not torch.equal(
    new_emb[0],
    old_row,
):
    fail(
        "new emb_g row 0 != source"
    )

if not torch.equal(
    new_emb[1],
    old_row,
):
    fail(
        "new emb_g row 1 != source"
    )

if not torch.equal(
    new_emb[0],
    new_emb[1],
):
    fail(
        "new emb_g rows are not identical"
    )


#
# Make pretrain checkpoints.
#
# We intentionally remove old optimizer states and reset
# iteration metadata. These are PRETRAIN files, never resume files.
#
new_g_model = {
    key:
        (
            new_emb
            if key == "emb_g.weight"
            else value.detach().cpu().clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
        )
    for key, value
    in state_g.items()
}


def make_pretrain(
    source_ckpt,
    model_state,
):
    return {
        "model":
            model_state,

        "optimizer":
            None,

        "learning_rate":
            float(
                source_ckpt[
                    "learning_rate"
                ]
            ),

        "iteration":
            0,
    }


new_g = make_pretrain(
    g,
    new_g_model,
)


new_d = make_pretrain(
    d,
    {
        key:
            value.detach().cpu().clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
        for key, value
        in d["model"].items()
    },
)


new_dur = make_pretrain(
    dur,
    {
        key:
            value.detach().cpu().clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
        for key, value
        in dur["model"].items()
    },
)


TMP.mkdir(
    parents=True,
    exist_ok=False,
)

PRE_G = (
    TMP / "pretrain_G.pth"
)

PRE_D = (
    TMP / "pretrain_D.pth"
)

PRE_DUR = (
    TMP / "pretrain_DUR.pth"
)


torch.save(
    new_g,
    PRE_G,
)

torch.save(
    new_d,
    PRE_D,
)

torch.save(
    new_dur,
    PRE_DUR,
)


#
# Reload from disk and verify exact prestart contract.
#
rg = load_ckpt(
    PRE_G
)

rd = load_ckpt(
    PRE_D
)

rdur = load_ckpt(
    PRE_DUR
)


for label, ckpt in [
    ("G", rg),
    ("D", rd),
    ("DUR", rdur),
]:
    if ckpt["optimizer"] is not None:
        fail(
            f"{label} pretrain optimizer not stripped"
        )

    if int(
        ckpt["iteration"]
    ) != 0:
        fail(
            f"{label} pretrain iteration != 0"
        )


reloaded_emb = rg[
    "model"
][
    "emb_g.weight"
]


if tuple(
    reloaded_emb.shape
) != (2, 256):
    fail(
        "reloaded emb_g shape != (2,256)"
    )

if not torch.equal(
    reloaded_emb[0],
    old_row,
):
    fail(
        "reloaded SID0 != E27 source row"
    )

if not torch.equal(
    reloaded_emb[1],
    old_row,
):
    fail(
        "reloaded SID1 != E27 source row"
    )

if not torch.equal(
    reloaded_emb[0],
    reloaded_emb[1],
):
    fail(
        "reloaded SID0 != SID1"
    )


#
# All other G tensors must be bitwise equal.
#
source_keys = set(
    state_g.keys()
)

new_keys = set(
    rg["model"].keys()
)

if source_keys != new_keys:
    fail(
        "generator key set changed"
    )


changed_keys = []

for key in sorted(
    source_keys
):
    if key == "emb_g.weight":
        continue

    a = state_g[key]
    b = rg["model"][key]

    if torch.is_tensor(a):
        if not torch.equal(
            a.detach().cpu(),
            b,
        ):
            changed_keys.append(
                key
            )
    elif a != b:
        changed_keys.append(
            key
        )


if changed_keys:
    fail(
        "unexpected changed G keys: "
        + repr(
            changed_keys[:20]
        )
    )


#
# D/DUR model states must be bitwise equal to E27.
#
for label, source, rebuilt in [
    (
        "D",
        d["model"],
        rd["model"],
    ),
    (
        "DUR",
        dur["model"],
        rdur["model"],
    ),
]:
    if set(source) != set(rebuilt):
        fail(
            f"{label} key set changed"
        )

    changed = []

    for key in source:
        a = source[key]
        b = rebuilt[key]

        if torch.is_tensor(a):
            if not torch.equal(
                a.detach().cpu(),
                b,
            ):
                changed.append(
                    key
                )
        elif a != b:
            changed.append(
                key
            )

    if changed:
        fail(
            f"{label} model state changed: "
            f"{changed[:20]}"
        )


#
# Base config.
#
with SRC_CONFIG.open(
    "r",
    encoding="utf-8",
) as f:
    config = json.load(
        f
    )


data = config[
    "data"
]

model = config[
    "model"
]


if bool(
    data.get(
        "disable_bert",
        False,
    )
) is not True:
    fail(
        "foundation disable_bert is not True"
    )

if int(
    data.get(
        "max_text_len",
        -1,
    )
) != 512:
    fail(
        "foundation max_text_len != 512"
    )


#
# No style-factorized training for this experiment.
#
existing_n_styles = model.get(
    "n_styles",
    0,
)

if existing_n_styles not in (
    None,
    0,
):
    fail(
        f"unexpected model.n_styles="
        f"{existing_n_styles}"
    )


data[
    "training_files"
] = str(
    (
        DATASET
        / "train.list"
    ).resolve()
)

data[
    "validation_files"
] = str(
    (
        DATASET
        / "val.list"
    ).resolve()
)

if "test_files" in data:
    data[
        "test_files"
    ] = str(
        (
            DATASET
            / "test.list"
        ).resolve()
    )


data[
    "spk2id"
] = SPEAKER_MAP

data[
    "n_speakers"
] = 2

data[
    "disable_bert"
] = True

data[
    "max_text_len"
] = 512


#
# Remove optional style mapping if inherited.
#
data.pop(
    "style2id",
    None,
)


#
# Pretrain paths point to the FINAL frozen location,
# not TMP paths.
#
config[
    "pretrain_G"
] = str(
    OUT
    / "pretrain_G.pth"
)

config[
    "pretrain_D"
] = str(
    OUT
    / "pretrain_D.pth"
)

config[
    "pretrain_dur"
] = str(
    OUT
    / "pretrain_DUR.pth"
)


#
# IMPORTANT:
# throughput fields are NOT finalized here.
# They remain inherited solely as a benchmark base.
#
(
    TMP / "config.base.json"
).write_text(
    json.dumps(
        config,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


manifest = {
    "artifact":
        "dialogs_np_e27_prestart_v1",

    "purpose":
        (
            "fresh Dialogs neutral+positive "
            "adaptation prestart"
        ),

    "foundation": {
        "epoch":
            27,

        "global_step":
            56645,

        "G_sha256":
            EXPECTED_SHA["G"],

        "D_sha256":
            EXPECTED_SHA["D"],

        "DUR_sha256":
            EXPECTED_SHA["DUR"],
    },

    "speaker_map":
        SPEAKER_MAP,

    "generator_expansion": {
        "source_emb_g_shape":
            [1, 256],

        "target_emb_g_shape":
            [2, 256],

        "SID0":
            "exact E27 emb_g row 0",

        "SID1":
            "exact E27 emb_g row 0",

        "SID0_equals_SID1":
            True,

        "all_other_G_parameters":
            "bitwise identical to E27",
    },

    "pretrain_semantics": {
        "optimizer_state":
            "stripped from G/D/DUR",

        "checkpoint_iteration":
            0,

        "training_global_step":
            (
                "must start at 0 because "
                "new model_dir contains no "
                "resume checkpoints"
            ),
    },

    "dataset": {
        "artifact":
            str(DATASET),

        "train_rows":
            6433,

        "val_rows":
            43,

        "test_rows":
            55,

        "hours":
            9.983843676,

        "max_text_len":
            512,
    },

    "config": {
        "file":
            "config.base.json",

        "production_throughput_parameters":
            "NOT YET FROZEN",
    },
}


(
    TMP / "manifest.json"
).write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


with (
    TMP / "SHA256SUMS.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    for path in sorted(
        TMP.iterdir()
    ):
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS.txt"
        ):
            f.write(
                f"{sha256(path)}  "
                f"{path.name}\n"
            )


#
# Atomic freeze.
#
TMP.rename(
    OUT
)


print()
print(
    "===== PRESTART CONTRACT =====",
    flush=True,
)

print(
    "source E27 emb_g:",
    tuple(
        old_emb.shape
    ),
    flush=True,
)

print(
    "new emb_g:",
    tuple(
        reloaded_emb.shape
    ),
    flush=True,
)

print(
    "SID0 == E27 row0:",
    torch.equal(
        reloaded_emb[0],
        old_row,
    ),
    flush=True,
)

print(
    "SID1 == E27 row0:",
    torch.equal(
        reloaded_emb[1],
        old_row,
    ),
    flush=True,
)

print(
    "SID0 == SID1:",
    torch.equal(
        reloaded_emb[0],
        reloaded_emb[1],
    ),
    flush=True,
)

print(
    "other G tensors changed:",
    len(
        changed_keys
    ),
    flush=True,
)

print(
    "G optimizer:",
    rg["optimizer"],
    flush=True,
)

print(
    "D optimizer:",
    rd["optimizer"],
    flush=True,
)

print(
    "DUR optimizer:",
    rdur["optimizer"],
    flush=True,
)

print(
    "pretrain iteration:",
    rg["iteration"],
    rd["iteration"],
    rdur["iteration"],
    flush=True,
)

print()
print(
    "pretrain_G SHA256:",
    sha256(
        OUT / "pretrain_G.pth"
    ),
    flush=True,
)

print(
    "pretrain_D SHA256:",
    sha256(
        OUT / "pretrain_D.pth"
    ),
    flush=True,
)

print(
    "pretrain_DUR SHA256:",
    sha256(
        OUT / "pretrain_DUR.pth"
    ),
    flush=True,
)

print(
    "config.base SHA256:",
    sha256(
        OUT / "config.base.json"
    ),
    flush=True,
)

print()
print(
    "frozen:",
    OUT,
    flush=True,
)

print()
print(
    "DIALOGS NP E27 PRESTART V1: PASS",
    flush=True,
)
