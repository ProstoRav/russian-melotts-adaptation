from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = (
    Path.home()
    / "projects/ru_melotts_research"
)

SOURCE = (
    PROJECT
    / "artifacts/c2_yocpt_text_catalog_v1"
    / "candidates_text.parquet"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_f30_clean_pool_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Hard filter configuration
# ---------------------------------------------------------

MIN_WORDS = 6

MIN_CPS = 7.0
MAX_CPS = 19.5

MIN_WPS = 1.0
MAX_WPS = 3.6

MAX_LATIN_LETTER_RATIO = 0.25

RATE_CENTER_CPS = 13.480885880966282
RATE_CENTER_WPS = 2.1963824938904466


SPACE_RE = re.compile(
    r"\s+"
)

NON_WORD_RE = re.compile(
    r"[^0-9a-zа-яё]+",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------
# Text helpers
# ---------------------------------------------------------

def normalize_space(value) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(value),
    )

    return SPACE_RE.sub(
        " ",
        text,
    ).strip()


def canonical_text(value) -> str:
    text = normalize_space(
        value
    ).lower()

    text = text.replace(
        "+",
        "",
    )

    text = text.replace(
        "ё",
        "е",
    )

    text = NON_WORD_RE.sub(
        " ",
        text,
    )

    return SPACE_RE.sub(
        " ",
        text,
    ).strip()


def make_training_text(value) -> str:
    """
    Keep punctuation/case/ё from text_denorm,
    but remove YO-CPT stress markup.

    Raw text_denorm remains untouched in catalog.
    """

    text = normalize_space(
        value
    )

    text = text.replace(
        "+",
        "",
    )

    # Rare alternative acute stress marks.
    text = text.replace(
        "\u0301",
        "",
    )

    return unicodedata.normalize(
        "NFC",
        SPACE_RE.sub(
            " ",
            text,
        ).strip(),
    )


def is_cyrillic_letter(ch: str) -> bool:
    cp = ord(ch)

    return (
        0x0400 <= cp <= 0x052F
        or
        0x2DE0 <= cp <= 0x2DFF
        or
        0xA640 <= cp <= 0xA69F
    )


def is_latin_letter(ch: str) -> bool:
    cp = ord(ch)

    return (
        0x0041 <= cp <= 0x005A
        or
        0x0061 <= cp <= 0x007A
        or
        0x00C0 <= cp <= 0x024F
    )


def text_script_stats(text: str):
    letters = 0
    latin = 0
    unsupported_letters = []

    for ch in text:

        if not unicodedata.category(
            ch
        ).startswith("L"):
            continue

        letters += 1

        if is_latin_letter(ch):
            latin += 1

        elif is_cyrillic_letter(ch):
            pass

        else:
            unsupported_letters.append(
                ch
            )

    latin_ratio = (
        latin / letters
        if letters
        else 0.0
    )

    return (
        letters,
        latin,
        latin_ratio,
        unsupported_letters,
    )


def percentile(values, q):
    if not values:
        return None

    vals = sorted(values)

    pos = (
        (len(vals) - 1)
        * q
    )

    lo = int(
        math.floor(pos)
    )

    hi = int(
        math.ceil(pos)
    )

    if lo == hi:
        return vals[lo]

    frac = pos - lo

    return (
        vals[lo]
        * (1.0 - frac)
        +
        vals[hi]
        * frac
    )


def quantiles(values):
    return {
        "p01":
            percentile(
                values,
                0.01,
            ),

        "p10":
            percentile(
                values,
                0.10,
            ),

        "p25":
            percentile(
                values,
                0.25,
            ),

        "p50":
            percentile(
                values,
                0.50,
            ),

        "p75":
            percentile(
                values,
                0.75,
            ),

        "p90":
            percentile(
                values,
                0.90,
            ),

        "p99":
            percentile(
                values,
                0.99,
            ),
    }


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

if not SOURCE.exists():
    raise RuntimeError(
        f"Missing source catalog: {SOURCE}"
    )

table = pq.read_table(
    SOURCE
)

rows = table.to_pylist()

print(
    "source rows:",
    len(rows),
)

source_seconds = sum(
    float(r["duration"])
    for r in rows
)

print(
    "source hours:",
    source_seconds / 3600.0,
)


# ---------------------------------------------------------
# Hard quality filtering
# ---------------------------------------------------------

accepted = []

rejection_any = Counter()

rejection_examples = defaultdict(
    list
)

for row in rows:

    duration = float(
        row["duration"]
    )

    text = normalize_space(
        row.get("text")
    )

    denorm = normalize_space(
        row.get("text_denorm")
    )

    training_text = (
        make_training_text(
            denorm
        )
    )

    words = len(
        text.split()
    )

    chars = len(
        text
    )

    cps = (
        chars / duration
        if duration > 0
        else 0.0
    )

    wps = (
        words / duration
        if duration > 0
        else 0.0
    )

    (
        letter_count,
        latin_count,
        latin_ratio,
        unsupported_letters,
    ) = text_script_stats(
        training_text
    )

    reasons = []

    if not denorm:
        reasons.append(
            "missing_text_denorm"
        )

    if not training_text:
        reasons.append(
            "empty_training_text"
        )

    if words < MIN_WORDS:
        reasons.append(
            "too_few_words"
        )

    if cps < MIN_CPS:
        reasons.append(
            "cps_too_low"
        )

    if cps > MAX_CPS:
        reasons.append(
            "cps_too_high"
        )

    if wps < MIN_WPS:
        reasons.append(
            "wps_too_low"
        )

    if wps > MAX_WPS:
        reasons.append(
            "wps_too_high"
        )

    if (
        latin_ratio
        > MAX_LATIN_LETTER_RATIO
    ):
        reasons.append(
            "too_much_latin"
        )

    if unsupported_letters:
        reasons.append(
            "unsupported_letter_script"
        )

    canonical = canonical_text(
        text
    )

    if not canonical:
        reasons.append(
            "empty_canonical_text"
        )

    if reasons:

        for reason in reasons:

            rejection_any[
                reason
            ] += 1

            if (
                len(
                    rejection_examples[
                        reason
                    ]
                )
                < 20
            ):
                rejection_examples[
                    reason
                ].append({
                    "utt_id":
                        row["utt_id"],

                    "duration":
                        duration,

                    "mos_score":
                        float(
                            row["mos_score"]
                        ),

                    "cps":
                        cps,

                    "wps":
                        wps,

                    "latin_ratio":
                        latin_ratio,

                    "text":
                        text,

                    "text_denorm":
                        denorm,
                })

        continue

    #
    # Mild diagnostic quality score.
    # Not yet the final F30 selection score.
    #
    mos = float(
        row["mos_score"]
    )

    rate_penalty = (
        abs(
            cps
            - RATE_CENTER_CPS
        )
        / RATE_CENTER_CPS
        +
        abs(
            wps
            - RATE_CENTER_WPS
        )
        / RATE_CENTER_WPS
    )

    accepted.append({
        **row,

        "training_text":
            training_text,

        "canonical_text":
            canonical,

        "word_count":
            int(words),

        "char_count":
            int(chars),

        "chars_per_sec":
            float(cps),

        "words_per_sec":
            float(wps),

        "latin_letter_ratio":
            float(
                latin_ratio
            ),

        "quality_score":
            float(
                mos
                - 0.10
                * rate_penalty
            ),
    })


hard_filter_seconds = sum(
    float(r["duration"])
    for r in accepted
)

print()
print(
    "after hard filters rows:",
    len(accepted),
)

print(
    "after hard filters hours:",
    hard_filter_seconds
    / 3600.0,
)


# ---------------------------------------------------------
# Canonical exact-text dedup
#
# Keep best MOS occurrence.
# Tie-break toward normal speech rate,
# then deterministic utt_id.
# ---------------------------------------------------------

groups = defaultdict(list)

for row in accepted:
    groups[
        row["canonical_text"]
    ].append(row)


deduped = []

duplicate_removed = []

for key, group in groups.items():

    ranked = sorted(
        group,
        key=lambda r: (
            -float(
                r["mos_score"]
            ),

            abs(
                float(
                    r["chars_per_sec"]
                )
                - RATE_CENTER_CPS
            ),

            abs(
                float(
                    r["words_per_sec"]
                )
                - RATE_CENTER_WPS
            ),

            r["utt_id"],
        ),
    )

    deduped.append(
        ranked[0]
    )

    duplicate_removed.extend(
        ranked[1:]
    )


deduped.sort(
    key=lambda r: (
        r["source_shard"],
        int(
            r["row_group"]
        ),
        int(
            r["row_in_group"]
        ),
    )
)


dedup_seconds = sum(
    float(r["duration"])
    for r in deduped
)

duplicate_removed_seconds = sum(
    float(r["duration"])
    for r in duplicate_removed
)

print()
print(
    "after dedup rows:",
    len(deduped),
)

print(
    "after dedup hours:",
    dedup_seconds / 3600.0,
)

print(
    "duplicates removed rows:",
    len(
        duplicate_removed
    ),
)

print(
    "duplicates removed hours:",
    duplicate_removed_seconds
    / 3600.0,
)


# ---------------------------------------------------------
# Save clean pool
# ---------------------------------------------------------

clean_path = (
    OUT
    / "clean_pool.parquet"
)

pq.write_table(
    pa.Table.from_pylist(
        deduped
    ),
    clean_path,
    compression="zstd",
)


# ---------------------------------------------------------
# Concentration/group diagnostics
# ---------------------------------------------------------

video_seconds = defaultdict(
    float
)

local_seconds = defaultdict(
    float
)

global_seconds = defaultdict(
    float
)

video_rows = Counter()
local_rows = Counter()
global_rows = Counter()

for r in deduped:

    duration = float(
        r["duration"]
    )

    video = (
        r.get(
            "source_video_id"
        )
        or "<missing>"
    )

    local = (
        r.get(
            "local_spk_id"
        )
        or "<missing>"
    )

    global_id = (
        r.get(
            "global_spk_id"
        )
    )

    video_seconds[
        video
    ] += duration

    video_rows[
        video
    ] += 1

    local_seconds[
        local
    ] += duration

    local_rows[
        local
    ] += 1

    if global_id:
        global_seconds[
            global_id
        ] += duration

        global_rows[
            global_id
        ] += 1


def capped_hours(
    values,
    cap_sec,
):
    return (
        sum(
            min(
                x,
                cap_sec,
            )
            for x in values
        )
        / 3600.0
    )


def group_threshold_counts(
    seconds_map,
):
    thresholds = [
        15,
        30,
        45,
        60,
        75,
        90,
        120,
    ]

    return {
        str(sec):
            sum(
                1
                for value
                in seconds_map.values()
                if value >= sec
            )
        for sec in thresholds
    }


def group_quantiles(
    seconds_map,
):
    values = list(
        seconds_map.values()
    )

    return {
        "groups":
            len(values),

        "seconds": {
            "p50":
                percentile(
                    values,
                    0.50,
                ),

            "p75":
                percentile(
                    values,
                    0.75,
                ),

            "p90":
                percentile(
                    values,
                    0.90,
                ),

            "p95":
                percentile(
                    values,
                    0.95,
                ),

            "p99":
                percentile(
                    values,
                    0.99,
                ),

            "max":
                max(
                    values,
                    default=0.0,
                ),
        },

        "groups_with_at_least_seconds":
            group_threshold_counts(
                seconds_map
            ),
    }


caps_sec = [
    60,
    90,
    120,
    180,
    300,
    600,
]


cap_feasibility = {
    "source_video": {
        str(x):
            capped_hours(
                video_seconds.values(),
                x,
            )
        for x in caps_sec
    },

    "local_speaker": {
        str(x):
            capped_hours(
                local_seconds.values(),
                x,
            )
        for x in caps_sec
    },

    "global_speaker_present_only": {
        str(x):
            capped_hours(
                global_seconds.values(),
                x,
            )
        for x in caps_sec
    },
}


# ---------------------------------------------------------
# Duration / MOS availability
# ---------------------------------------------------------

duration_bins = {
    "2.5-6":
        0.0,

    "6-8":
        0.0,

    "8-10":
        0.0,

    "10-12":
        0.0,
}


mos_bins = {
    "4.20-4.30":
        0.0,

    "4.30-4.40":
        0.0,

    "4.40-4.50":
        0.0,

    "4.50+":
        0.0,
}


cps_values = []
wps_values = []

for r in deduped:

    dur = float(
        r["duration"]
    )

    mos = float(
        r["mos_score"]
    )

    cps_values.append(
        float(
            r["chars_per_sec"]
        )
    )

    wps_values.append(
        float(
            r["words_per_sec"]
        )
    )

    if dur < 6.0:
        duration_bins[
            "2.5-6"
        ] += dur

    elif dur < 8.0:
        duration_bins[
            "6-8"
        ] += dur

    elif dur < 10.0:
        duration_bins[
            "8-10"
        ] += dur

    else:
        duration_bins[
            "10-12"
        ] += dur

    if mos < 4.30:
        mos_bins[
            "4.20-4.30"
        ] += dur

    elif mos < 4.40:
        mos_bins[
            "4.30-4.40"
        ] += dur

    elif mos < 4.50:
        mos_bins[
            "4.40-4.50"
        ] += dur

    else:
        mos_bins[
            "4.50+"
        ] += dur


duration_bins = {
    k:
        v / 3600.0
    for k, v
    in duration_bins.items()
}

mos_bins = {
    k:
        v / 3600.0
    for k, v
    in mos_bins.items()
}


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

summary = {
    "source": {
        "rows":
            len(rows),

        "hours":
            source_seconds
            / 3600.0,
    },

    "filters": {
        "min_words":
            MIN_WORDS,

        "cps":
            [
                MIN_CPS,
                MAX_CPS,
            ],

        "wps":
            [
                MIN_WPS,
                MAX_WPS,
            ],

        "max_latin_letter_ratio":
            MAX_LATIN_LETTER_RATIO,

        "unsupported_letter_scripts_allowed":
            False,

        "text_denorm_required":
            True,
    },

    "rejection_reason_counts_any": {
        k:
            int(v)
        for k, v
        in sorted(
            rejection_any.items()
        )
    },

    "after_hard_filters": {
        "rows":
            len(accepted),

        "hours":
            hard_filter_seconds
            / 3600.0,
    },

    "dedup": {
        "removed_rows":
            len(
                duplicate_removed
            ),

        "removed_hours":
            duplicate_removed_seconds
            / 3600.0,
    },

    "clean_pool": {
        "rows":
            len(deduped),

        "hours":
            dedup_seconds
            / 3600.0,

        "unique_source_videos":
            len(
                video_seconds
            ),

        "unique_local_speakers":
            len(
                local_seconds
            ),

        "unique_global_speakers_present":
            len(
                global_seconds
            ),
    },

    "speech_rate_after_cleaning": {
        "chars_per_sec":
            quantiles(
                cps_values
            ),

        "words_per_sec":
            quantiles(
                wps_values
            ),
    },

    "duration_bin_hours":
        duration_bins,

    "mos_bin_hours":
        mos_bins,

    "source_video_groups":
        group_quantiles(
            video_seconds
        ),

    "local_speaker_groups":
        group_quantiles(
            local_seconds
        ),

    "global_speaker_groups_present_only":
        group_quantiles(
            global_seconds
        ),

    "cap_feasibility_hours": {
        "cap_seconds":
            caps_sec,

        **cap_feasibility,
    },

    "clean_pool_path":
        str(
            clean_path
        ),
}


summary_path = (
    OUT
    / "summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------
# Save rejection examples
# ---------------------------------------------------------

(
    OUT
    / "rejection_examples.json"
).write_text(
    json.dumps(
        rejection_examples,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------
# Top group tables
# ---------------------------------------------------------

def save_group_table(
    filename,
    seconds_map,
    rows_map,
):
    items = sorted(
        seconds_map.items(),
        key=lambda x: (
            -x[1],
            str(x[0]),
        ),
    )

    with (
        OUT
        / filename
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "rank\tid\trows\thours\n"
        )

        for i, (
            key,
            seconds,
        ) in enumerate(
            items[:100],
            start=1,
        ):
            f.write(
                f"{i}\t"
                f"{key}\t"
                f"{rows_map[key]}\t"
                f"{seconds / 3600.0:.6f}\n"
            )


save_group_table(
    "top_source_videos.tsv",
    video_seconds,
    video_rows,
)

save_group_table(
    "top_local_speakers.tsv",
    local_seconds,
    local_rows,
)

save_group_table(
    "top_global_speakers.tsv",
    global_seconds,
    global_rows,
)


print()
print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print(
    "clean pool:",
    clean_path,
)

print()
print(
    "YO-CPT F30 CLEAN POOL V1: PASS"
)
