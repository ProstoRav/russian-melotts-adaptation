#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(
    os.environ.get(
        "PROJECT",
        "${PROJECT_ROOT}",
    )
)

METRICS = (
    PROJECT
    / "artifacts/eval/tts_trajectory_metrics_v2"
)

BOXPLOTS = (
    PROJECT
    / "artifacts/eval/tts_publication_boxplots_v3"
)

STYLE = (
    PROJECT
    / "artifacts/eval/dialogs_matched_style_defense_v1"
)

IDENTITY = (
    PROJECT
    / "artifacts/eval/speaker_identity_eval_v1"
)

IDENTITY_PLOTS = (
    PROJECT
    / "artifacts/eval/speaker_identity_publication_plots_v1"
)

OUT = (
    PROJECT
    / "artifacts/eval/final_evaluation_package_v1"
)

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 29842

CHOSEN = {
    "dialogs": 38,
    "yocpt": 27,
}

LATER = {
    "dialogs": 40,
    "yocpt": 30,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path):
    if not path.is_file():
        raise RuntimeError(
            f"required file missing: {path}"
        )


def require_complete(path: Path):
    require_file(
        path / ".complete"
    )


for directory in (
    METRICS,
    STYLE,
    IDENTITY,
    IDENTITY_PLOTS,
):
    require_complete(
        directory
    )

require_file(
    METRICS
    / "per_audio_metrics.csv"
)

require_file(
    METRICS
    / "aggregate_overall.csv"
)

require_file(
    STYLE
    / "summary_by_epoch_metric.csv"
)

require_file(
    STYLE
    / "embedding_separation.csv"
)

require_file(
    IDENTITY
    / "yocpt_speaker_similarity_summary.csv"
)

require_file(
    IDENTITY
    / "dialogs_defense_voice_consistency_summary.csv"
)


if OUT.exists():
    raise RuntimeError(
        f"output already exists: {OUT}"
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)

TABLES = OUT / "tables"
MAIN_PLOTS = OUT / "plots/main"
APPENDIX_PLOTS = OUT / "plots/appendix"

TABLES.mkdir(
    parents=True
)

MAIN_PLOTS.mkdir(
    parents=True
)

APPENDIX_PLOTS.mkdir(
    parents=True
)


per_audio = pd.read_csv(
    METRICS
    / "per_audio_metrics.csv"
)

aggregate = pd.read_csv(
    METRICS
    / "aggregate_overall.csv"
)

style = pd.read_csv(
    STYLE
    / "summary_by_epoch_metric.csv"
)

embedding_sep = pd.read_csv(
    STYLE
    / "embedding_separation.csv"
)

speaker = pd.read_csv(
    IDENTITY
    / "yocpt_speaker_similarity_summary.csv"
)

voice = pd.read_csv(
    IDENTITY
    / "dialogs_defense_voice_consistency_summary.csv"
)


# ============================================================
# Corpus metric bootstrap
# ============================================================

def corpus_metric(
    df,
    edits_col,
    units_col,
):
    denominator = float(
        df[
            units_col
        ].sum()
    )

    if denominator <= 0:
        raise RuntimeError(
            f"zero denominator for {units_col}"
        )

    return float(
        df[
            edits_col
        ].sum()
        / denominator
    )


def bootstrap_corpus_metric(
    df,
    edits_col,
    units_col,
    seed,
):
    edits = df[
        edits_col
    ].to_numpy(
        dtype=float
    )

    units = df[
        units_col
    ].to_numpy(
        dtype=float
    )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(df),
        size=(
            BOOTSTRAP_SAMPLES,
            len(df),
        ),
    )

    numer = edits[
        idx
    ].sum(
        axis=1
    )

    denom = units[
        idx
    ].sum(
        axis=1
    )

    values = (
        numer
        / denom
    )

    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


def paired_relative_reduction_bootstrap(
    first,
    chosen,
    edits_col,
    units_col,
    seed,
):
    columns = [
        "row_id",
        edits_col,
        units_col,
    ]

    a = first[
        columns
    ].rename(
        columns={
            edits_col:
                "first_edits",
            units_col:
                "first_units",
        }
    )

    b = chosen[
        columns
    ].rename(
        columns={
            edits_col:
                "chosen_edits",
            units_col:
                "chosen_units",
        }
    )

    merged = a.merge(
        b,
        on="row_id",
        how="inner",
        validate="one_to_one",
    )

    if (
        len(merged)
        != len(first)
        or
        len(merged)
        != len(chosen)
    ):
        raise RuntimeError(
            "paired bootstrap row mismatch"
        )

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        len(merged),
        size=(
            BOOTSTRAP_SAMPLES,
            len(merged),
        ),
    )

    first_metric = (
        merged[
            "first_edits"
        ].to_numpy()[
            idx
        ].sum(
            axis=1
        )
        /
        merged[
            "first_units"
        ].to_numpy()[
            idx
        ].sum(
            axis=1
        )
    )

    chosen_metric = (
        merged[
            "chosen_edits"
        ].to_numpy()[
            idx
        ].sum(
            axis=1
        )
        /
        merged[
            "chosen_units"
        ].to_numpy()[
            idx
        ].sum(
            axis=1
        )
    )

    valid = (
        first_metric > 0
    )

    reduction = (
        (
            first_metric[
                valid
            ]
            -
            chosen_metric[
                valid
            ]
        )
        /
        first_metric[
            valid
        ]
        * 100.0
    )

    delta_pp = (
        (
            chosen_metric
            -
            first_metric
        )
        * 100.0
    )

    return {
        "relative_reduction_ci95_low":
            float(
                np.percentile(
                    reduction,
                    2.5,
                )
            ),

        "relative_reduction_ci95_high":
            float(
                np.percentile(
                    reduction,
                    97.5,
                )
            ),

        "delta_pp_ci95_low":
            float(
                np.percentile(
                    delta_pp,
                    2.5,
                )
            ),

        "delta_pp_ci95_high":
            float(
                np.percentile(
                    delta_pp,
                    97.5,
                )
            ),
    }


bootstrap_rows = []

metric_defs = {
    "WER": (
        "word_edits",
        "word_ref_units",
    ),
    "CER": (
        "char_edits",
        "char_ref_units",
    ),
}


for experiment_index, experiment in enumerate(
    [
        "dialogs",
        "yocpt",
    ]
):

    x = per_audio[
        per_audio[
            "experiment"
        ]
        == experiment
    ]

    epoch_values = sorted(
        int(v)
        for v in x[
            "epoch"
        ].dropna().unique()
    )

    groups = [
        (
            "Natural reference",
            x[
                x[
                    "checkpoint_role"
                ]
                == "REFERENCE"
            ],
            None,
        )
    ]

    for epoch in epoch_values:
        groups.append(
            (
                (
                    "Chosen candidate"
                    if epoch
                    == CHOSEN[
                        experiment
                    ]
                    else "Epoch"
                ),
                x[
                    x[
                        "epoch"
                    ]
                    == epoch
                ],
                epoch,
            )
        )


    for group_index, (
        role,
        group,
        epoch,
    ) in enumerate(
        groups
    ):

        for metric_index, (
            metric,
            (
                edits_col,
                units_col,
            ),
        ) in enumerate(
            metric_defs.items()
        ):

            estimate = (
                corpus_metric(
                    group,
                    edits_col,
                    units_col,
                )
            )

            ci_low, ci_high = (
                bootstrap_corpus_metric(
                    group,
                    edits_col,
                    units_col,
                    (
                        BOOTSTRAP_SEED
                        + experiment_index
                        * 1000
                        + group_index
                        * 10
                        + metric_index
                    ),
                )
            )

            bootstrap_rows.append(
                {
                    "experiment":
                        experiment,

                    "holdout_kind":
                        (
                            "test"
                            if experiment
                            == "dialogs"
                            else "validation"
                        ),

                    "role":
                        role,

                    "epoch":
                        epoch,

                    "metric":
                        metric,

                    "n":
                        int(
                            len(
                                group
                            )
                        ),

                    "corpus_value":
                        estimate,

                    "bootstrap_ci95_low":
                        ci_low,

                    "bootstrap_ci95_high":
                        ci_high,
                }
            )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df.to_csv(
    TABLES
    / "corpus_cer_wer_bootstrap_ci.csv",
    index=False,
)


# ============================================================
# E1 -> chosen paired bootstrap
# ============================================================

improvement_rows = []


for experiment_index, experiment in enumerate(
    [
        "dialogs",
        "yocpt",
    ]
):

    x = per_audio[
        per_audio[
            "experiment"
        ]
        == experiment
    ]

    first = x[
        x[
            "epoch"
        ]
        == 1
    ]

    chosen = x[
        x[
            "epoch"
        ]
        == CHOSEN[
            experiment
        ]
    ]


    for metric_index, (
        metric,
        (
            edits_col,
            units_col,
        ),
    ) in enumerate(
        metric_defs.items()
    ):

        first_value = (
            corpus_metric(
                first,
                edits_col,
                units_col,
            )
        )

        chosen_value = (
            corpus_metric(
                chosen,
                edits_col,
                units_col,
            )
        )

        reduction = (
            (
                first_value
                -
                chosen_value
            )
            /
            first_value
            * 100.0
        )

        ci = (
            paired_relative_reduction_bootstrap(
                first,
                chosen,
                edits_col,
                units_col,
                (
                    BOOTSTRAP_SEED
                    + 5000
                    + experiment_index
                    * 100
                    + metric_index
                ),
            )
        )

        improvement_rows.append(
            {
                "experiment":
                    experiment,

                "metric":
                    metric,

                "first_epoch":
                    1,

                "chosen_epoch":
                    CHOSEN[
                        experiment
                    ],

                "first_value":
                    first_value,

                "chosen_value":
                    chosen_value,

                "relative_reduction_pct":
                    reduction,

                **ci,
            }
        )


improvement_df = pd.DataFrame(
    improvement_rows
)

improvement_df.to_csv(
    TABLES
    / "first_to_chosen_bootstrap.csv",
    index=False,
)


# ============================================================
# Chosen -> later checkpoint context
# ============================================================

later_rows = []


for experiment in (
    "dialogs",
    "yocpt",
):

    chosen_epoch = (
        CHOSEN[
            experiment
        ]
    )

    later_epoch = (
        LATER[
            experiment
        ]
    )

    for metric in (
        "corpus_wer",
        "corpus_cer",
        "duration_mape_mean",
    ):

        chosen_row = aggregate[
            (
                aggregate[
                    "experiment"
                ]
                == experiment
            )
            &
            (
                aggregate[
                    "epoch"
                ]
                == chosen_epoch
            )
        ].iloc[0]

        later_row = aggregate[
            (
                aggregate[
                    "experiment"
                ]
                == experiment
            )
            &
            (
                aggregate[
                    "epoch"
                ]
                == later_epoch
            )
        ].iloc[0]

        chosen_value = float(
            chosen_row[
                metric
            ]
        )

        later_value = float(
            later_row[
                metric
            ]
        )

        later_rows.append(
            {
                "experiment":
                    experiment,

                "metric":
                    metric,

                "chosen_epoch":
                    chosen_epoch,

                "later_epoch":
                    later_epoch,

                "chosen_value":
                    chosen_value,

                "later_value":
                    later_value,

                "later_minus_chosen":
                    (
                        later_value
                        -
                        chosen_value
                    ),

                "relative_change_pct":
                    (
                        (
                            later_value
                            -
                            chosen_value
                        )
                        /
                        chosen_value
                        * 100.0
                        if chosen_value
                        != 0
                        else np.nan
                    ),
            }
        )


pd.DataFrame(
    later_rows
).to_csv(
    TABLES
    / "chosen_vs_later_checkpoint.csv",
    index=False,
)


# ============================================================
# Numeral sensitivity audit
# ============================================================

numeric_rows = []


for experiment in (
    "dialogs",
    "yocpt",
):

    refs = per_audio[
        (
            per_audio[
                "experiment"
            ]
            == experiment
        )
        &
        (
            per_audio[
                "checkpoint_role"
            ]
            == "REFERENCE"
        )
    ]

    contains_digit = (
        refs[
            "text"
        ]
        .astype(
            str
        )
        .str.contains(
            r"\d",
            regex=True,
        )
    )

    numeric_rows.append(
        {
            "experiment":
                experiment,

            "holdout_rows":
                int(
                    len(
                        refs
                    )
                ),

            "rows_with_digits":
                int(
                    contains_digit.sum()
                ),

            "fraction_with_digits":
                float(
                    contains_digit.mean()
                ),
        }
    )


numeric_df = pd.DataFrame(
    numeric_rows
)

numeric_df.to_csv(
    TABLES
    / "numeric_orthography_audit.csv",
    index=False,
)


# ============================================================
# Core chosen-candidate results
# ============================================================

main_rows = []


for experiment in (
    "dialogs",
    "yocpt",
):

    for metric in (
        "WER",
        "CER",
    ):

        r = improvement_df[
            (
                improvement_df[
                    "experiment"
                ]
                == experiment
            )
            &
            (
                improvement_df[
                    "metric"
                ]
                == metric
            )
        ].iloc[0]

        main_rows.append(
            {
                "experiment":
                    experiment,

                "dimension":
                    "Intelligibility",

                "metric":
                    f"Corpus {metric}",

                "first_epoch":
                    1,

                "first_value":
                    float(
                        r[
                            "first_value"
                        ]
                    ),

                "chosen_epoch":
                    int(
                        r[
                            "chosen_epoch"
                        ]
                    ),

                "chosen_value":
                    float(
                        r[
                            "chosen_value"
                        ]
                    ),

                "relative_change":
                    (
                        -float(
                            r[
                                "relative_reduction_pct"
                            ]
                        )
                    ),

                "relative_change_unit":
                    "%",

                "supporting_stat":
                    (
                        "paired bootstrap "
                        "relative reduction "
                        f"95% CI "
                        f"[{r['relative_reduction_ci95_low']:.2f}, "
                        f"{r['relative_reduction_ci95_high']:.2f}]%"
                    ),
            }
        )


# Dialogs style results at E38
for metric_name, display in (
    (
        "delta_duration_pct",
        "Positive − neutral duration",
    ),
    (
        "delta_rms_db",
        "Positive − neutral RMS",
    ),
):

    r = style[
        (
            style[
                "epoch"
            ]
            == 38
        )
        &
        (
            style[
                "metric"
            ]
            == metric_name
        )
    ].iloc[0]

    main_rows.append(
        {
            "experiment":
                "dialogs",

            "dimension":
                "Style separation",

            "metric":
                display,

            "first_epoch":
                np.nan,

            "first_value":
                np.nan,

            "chosen_epoch":
                38,

            "chosen_value":
                float(
                    r[
                        "median"
                    ]
                ),

            "relative_change":
                np.nan,

            "relative_change_unit":
                r[
                    "unit"
                ],

            "supporting_stat":
                (
                    f"median 95% CI "
                    f"[{r['bootstrap_median_ci95_low']:.3f}, "
                    f"{r['bootstrap_median_ci95_high']:.3f}], "
                    f"direction consistency "
                    f"{r['direction_consistency'] * 100:.1f}%"
                ),
        }
    )


emb_e1 = embedding_sep[
    embedding_sep[
        "epoch"
    ]
    == 1
].iloc[0]

emb_e38 = embedding_sep[
    embedding_sep[
        "epoch"
    ]
    == 38
].iloc[0]

main_rows.append(
    {
        "experiment":
            "dialogs",

        "dimension":
            "Style separation",

        "metric":
            "SID embedding L2 distance",

        "first_epoch":
            1,

        "first_value":
            float(
                emb_e1[
                    "l2_distance"
                ]
            ),

        "chosen_epoch":
            38,

        "chosen_value":
            float(
                emb_e38[
                    "l2_distance"
                ]
            ),

        "relative_change":
            (
                (
                    float(
                        emb_e38[
                            "l2_distance"
                        ]
                    )
                    /
                    float(
                        emb_e1[
                            "l2_distance"
                        ]
                    )
                )
                - 1.0
            )
            * 100.0,

        "relative_change_unit":
            "%",

        "supporting_stat":
            (
                "internal conditioning-space "
                "diagnostic"
            ),
    }
)


# YO-CPT speaker preservation
for epoch in (
    1,
    27,
):

    r = speaker[
        (
            speaker[
                "epoch"
            ]
            == epoch
        )
        &
        (
            speaker[
                "metric"
            ]
            ==
            "similarity_to_enrollment_centroid"
        )
    ].iloc[0]

    if epoch == 1:
        speaker_e1 = r
    else:
        speaker_chosen = r


natural_speaker = speaker[
    (
        speaker[
            "checkpoint"
        ]
        == "NATURAL_REFERENCE"
    )
    &
    (
        speaker[
            "metric"
        ]
        ==
        "similarity_to_enrollment_centroid"
    )
].iloc[0]


main_rows.append(
    {
        "experiment":
            "yocpt",

        "dimension":
            "Speaker identity",

        "metric":
            "Similarity to spk_298 enrollment centroid",

        "first_epoch":
            1,

        "first_value":
            float(
                speaker_e1[
                    "median"
                ]
            ),

        "chosen_epoch":
            27,

        "chosen_value":
            float(
                speaker_chosen[
                    "median"
                ]
            ),

        "relative_change":
            (
                float(
                    speaker_chosen[
                        "median"
                    ]
                )
                -
                float(
                    speaker_e1[
                        "median"
                    ]
                )
            ),

        "relative_change_unit":
            "cosine",

        "supporting_stat":
            (
                "chosen median 95% CI "
                f"[{speaker_chosen['median_ci95_low']:.6f}, "
                f"{speaker_chosen['median_ci95_high']:.6f}]; "
                "natural reference median "
                f"{natural_speaker['median']:.6f}"
            ),
    }
)


# Dialogs voice consistency
voice_e38 = voice[
    (
        voice[
            "epoch"
        ]
        == 38
    )
    &
    (
        voice[
            "metric"
        ]
        == "sid0_sid1_cosine"
    )
].iloc[0]

voice_e1 = voice[
    (
        voice[
            "epoch"
        ]
        == 1
    )
    &
    (
        voice[
            "metric"
        ]
        == "sid0_sid1_cosine"
    )
].iloc[0]


main_rows.append(
    {
        "experiment":
            "dialogs",

        "dimension":
            "Voice identity",

        "metric":
            "SID0 ↔ SID1 cosine similarity",

        "first_epoch":
            1,

        "first_value":
            float(
                voice_e1[
                    "median"
                ]
            ),

        "chosen_epoch":
            38,

        "chosen_value":
            float(
                voice_e38[
                    "median"
                ]
            ),

        "relative_change":
            (
                float(
                    voice_e38[
                        "median"
                    ]
                )
                -
                float(
                    voice_e1[
                        "median"
                    ]
                )
            ),

        "relative_change_unit":
            "cosine",

        "supporting_stat":
            (
                "chosen median 95% CI "
                f"[{voice_e38['median_ci95_low']:.6f}, "
                f"{voice_e38['median_ci95_high']:.6f}], "
                "matched text + seed"
            ),
    }
)


main_df = pd.DataFrame(
    main_rows
)

main_df.to_csv(
    TABLES
    / "main_results.csv",
    index=False,
)


# ============================================================
# Copy publication figures
# ============================================================

main_plot_sources = [
    (
        BOXPLOTS
        / "dialogs_wer_boxplot.png",
        "01_dialogs_wer.png",
    ),
    (
        BOXPLOTS
        / "dialogs_cer_boxplot.png",
        "02_dialogs_cer.png",
    ),
    (
        BOXPLOTS
        / "yocpt_wer_boxplot.png",
        "03_yocpt_wer.png",
    ),
    (
        BOXPLOTS
        / "yocpt_cer_boxplot.png",
        "04_yocpt_cer.png",
    ),
    (
        STYLE
        / "plots/duration_delta_boxplot.png",
        "05_dialogs_style_duration.png",
    ),
    (
        STYLE
        / "plots/rms_delta_boxplot.png",
        "06_dialogs_style_rms.png",
    ),
    (
        IDENTITY_PLOTS
        / "yocpt_enrollment_centroid_similarity.png",
        "07_yocpt_speaker_similarity.png",
    ),
    (
        IDENTITY_PLOTS
        / "dialogs_sid0_sid1_voice_consistency.png",
        "08_dialogs_voice_identity_consistency.png",
    ),
]


appendix_plot_sources = [
    (
        BOXPLOTS
        / "dialogs_duration_mape_boxplot.png",
        "dialogs_duration_mape.png",
    ),
    (
        BOXPLOTS
        / "dialogs_f0_mape_boxplot.png",
        "dialogs_f0_mape.png",
    ),
    (
        BOXPLOTS
        / "yocpt_duration_mape_boxplot.png",
        "yocpt_duration_mape.png",
    ),
    (
        BOXPLOTS
        / "yocpt_f0_mape_boxplot.png",
        "yocpt_f0_mape.png",
    ),
    (
        STYLE
        / "plots/f0_median_delta_boxplot.png",
        "dialogs_style_f0_median.png",
    ),
    (
        STYLE
        / "plots/f0_range_delta_boxplot.png",
        "dialogs_style_f0_range.png",
    ),
    (
        STYLE
        / "plots/voiced_fraction_delta_boxplot.png",
        "dialogs_style_voiced_fraction.png",
    ),
    (
        IDENTITY_PLOTS
        / "yocpt_matched_reference_similarity.png",
        "yocpt_matched_reference_similarity.png",
    ),
    (
        IDENTITY_PLOTS
        / "dialogs_actor_affinity_margin_appendix.png",
        "dialogs_actor_affinity_margin.png",
    ),
]


def copy_plot_set(
    sources,
    target_dir,
):
    for source_png, target_name in sources:
        require_file(
            source_png
        )

        stem = (
            source_png
            .with_suffix("")
        )

        for extension in (
            ".png",
            ".svg",
            ".pdf",
        ):

            source = (
                stem.with_suffix(
                    extension
                )
            )

            require_file(
                source
            )

            destination = (
                target_dir
                / Path(
                    target_name
                ).with_suffix(
                    extension
                ).name
            )

            shutil.copy2(
                source,
                destination,
            )


copy_plot_set(
    main_plot_sources,
    MAIN_PLOTS,
)

copy_plot_set(
    appendix_plot_sources,
    APPENDIX_PLOTS,
)


# ============================================================
# Human-readable README
# ============================================================

def pct(value):
    return (
        f"{value * 100:.2f}%"
    )


dialogs_wer = improvement_df[
    (
        improvement_df[
            "experiment"
        ]
        == "dialogs"
    )
    &
    (
        improvement_df[
            "metric"
        ]
        == "WER"
    )
].iloc[0]

dialogs_cer = improvement_df[
    (
        improvement_df[
            "experiment"
        ]
        == "dialogs"
    )
    &
    (
        improvement_df[
            "metric"
        ]
        == "CER"
    )
].iloc[0]

yocpt_wer = improvement_df[
    (
        improvement_df[
            "experiment"
        ]
        == "yocpt"
    )
    &
    (
        improvement_df[
            "metric"
        ]
        == "WER"
    )
].iloc[0]

yocpt_cer = improvement_df[
    (
        improvement_df[
            "experiment"
        ]
        == "yocpt"
    )
    &
    (
        improvement_df[
            "metric"
        ]
        == "CER"
    )
].iloc[0]


readme = f"""# Final Evaluation Package V1

## Scope

This package consolidates the objective evaluation trajectory:

- English-v3 pretraining → Russian YO-CPT adaptation → E27 chosen candidate.
- E27 initialization → Dialogs neutral/positive adaptation → E38 chosen candidate.

YO-CPT uses the frozen **validation holdout** (n=194).
Dialogs uses the frozen **test holdout** (n=55).

`E27` and `E38` are called **chosen candidate**. E30/E40 are shown simply as later measured epochs.

## Primary intelligibility results

### Dialogs

- Corpus WER: {pct(dialogs_wer["first_value"])} at E1 → {pct(dialogs_wer["chosen_value"])} at E38.
- Relative reduction: {dialogs_wer["relative_reduction_pct"]:.2f}%.
- Bootstrap 95% CI for relative reduction:
  [{dialogs_wer["relative_reduction_ci95_low"]:.2f}%, {dialogs_wer["relative_reduction_ci95_high"]:.2f}%].

- Corpus CER: {pct(dialogs_cer["first_value"])} at E1 → {pct(dialogs_cer["chosen_value"])} at E38.
- Relative reduction: {dialogs_cer["relative_reduction_pct"]:.2f}%.
- Bootstrap 95% CI for relative reduction:
  [{dialogs_cer["relative_reduction_ci95_low"]:.2f}%, {dialogs_cer["relative_reduction_ci95_high"]:.2f}%].

### YO-CPT

- Corpus WER: {pct(yocpt_wer["first_value"])} at E1 → {pct(yocpt_wer["chosen_value"])} at E27.
- Relative reduction: {yocpt_wer["relative_reduction_pct"]:.2f}%.
- Bootstrap 95% CI for relative reduction:
  [{yocpt_wer["relative_reduction_ci95_low"]:.2f}%, {yocpt_wer["relative_reduction_ci95_high"]:.2f}%].

- Corpus CER: {pct(yocpt_cer["first_value"])} at E1 → {pct(yocpt_cer["chosen_value"])} at E27.
- Relative reduction: {yocpt_cer["relative_reduction_pct"]:.2f}%.
- Bootstrap 95% CI for relative reduction:
  [{yocpt_cer["relative_reduction_ci95_low"]:.2f}%, {yocpt_cer["relative_reduction_ci95_high"]:.2f}%].

## Style separation

At Dialogs E38, controlled matched-text / matched-seed evaluation shows:

- positive − neutral duration median:
  {style[(style["epoch"] == 38) & (style["metric"] == "delta_duration_pct")].iloc[0]["median"]:.3f}%,
  direction consistency 9/9 prompts;

- positive − neutral RMS median:
  {style[(style["epoch"] == 38) & (style["metric"] == "delta_rms_db")].iloc[0]["median"]:.3f} dB,
  direction consistency 9/9 prompts.

Pitch-based features are not consistently directional at E38 and therefore are supporting diagnostics only.

The learned SID0/SID1 embedding L2 distance increased from
{emb_e1["l2_distance"]:.6f} at E1 to {emb_e38["l2_distance"]:.6f} at E38.

## Speaker / voice identity

### YO-CPT

The natural validation median similarity to the spk_298 train enrollment centroid is
{natural_speaker["median"]:.6f}.

The synthetic trajectory reaches:

- E1: {speaker_e1["median"]:.6f}
- E27 chosen candidate: {speaker_chosen["median"]:.6f}

This supports speaker preservation for the single-speaker YO-CPT stage.

### Dialogs

Dialogs contains two real source actors (M/S), but the adaptation labels encode neutral/positive rather than actor identity.
Therefore M/S affinity is treated as an appendix diagnostic, not speaker preservation.

For controlled S01–S09 matched prompts, SID0 neutral vs SID1 positive median external speaker-embedding cosine is
{voice_e38["median"]:.6f} at E38
(95% CI [{voice_e38["median_ci95_low"]:.6f}, {voice_e38["median_ci95_high"]:.6f}]).

This supports common synthetic voice identity across the two conditioning labels while their acoustic realization differs.

## Numeric orthography audit

No digit-containing reference texts occur in the current frozen evaluation holdouts:

- Dialogs: {int(numeric_df[numeric_df["experiment"] == "dialogs"].iloc[0]["rows_with_digits"])} / 55.
- YO-CPT: {int(numeric_df[numeric_df["experiment"] == "yocpt"].iloc[0]["rows_with_digits"])} / 194.

Therefore numeral normalization is not a contributor to the reported current holdout CER/WER.

## Interpretation limits

- CER/WER are ASR-mediated intelligibility metrics, not human MOS.
- Natural-reference ASR is a baseline, not an error floor to subtract from synthetic results.
- Dialogs F0 / energy / duration support style analysis but do not independently prove perceived emotion.
- Dialogs actor-centroid affinity is not a speaker-preservation metric because M/S identity was not used as conditioning.
- P10–P90 whiskers in publication figures are a presentation convention and are explicitly labeled as such.
"""


(
    OUT
    / "README.md"
).write_text(
    readme,
    encoding="utf-8",
)


# ============================================================
# Machine-readable summary
# ============================================================

summary = {
    "version": 1,

    "chosen_candidates": (
        CHOSEN
    ),

    "bootstrap": {
        "samples":
            BOOTSTRAP_SAMPLES,

        "seed":
            BOOTSTRAP_SEED,

        "unit":
            "utterance / row_id resampling",
    },

    "numeric_orthography": {
        row[
            "experiment"
        ]: {
            "holdout_rows":
                int(
                    row[
                        "holdout_rows"
                    ]
                ),

            "rows_with_digits":
                int(
                    row[
                        "rows_with_digits"
                    ]
                ),
        }
        for _, row in (
            numeric_df.iterrows()
        )
    },

    "source_artifacts": {
        "tts_trajectory_metrics_v2":
            str(
                METRICS
            ),

        "tts_publication_boxplots_v3":
            str(
                BOXPLOTS
            ),

        "dialogs_matched_style_defense_v1":
            str(
                STYLE
            ),

        "speaker_identity_eval_v1":
            str(
                IDENTITY
            ),

        "speaker_identity_publication_plots_v1":
            str(
                IDENTITY_PLOTS
            ),
    },
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


# ============================================================
# Checksums
# ============================================================

checksum_targets = sorted(
    path
    for path in OUT.rglob("*")
    if (
        path.is_file()
        and
        path.name
        not in {
            "SHA256SUMS.txt",
            ".complete",
        }
    )
)


with (
    OUT
    / "SHA256SUMS.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    for path in checksum_targets:

        relative = (
            path.relative_to(
                OUT
            )
        )

        f.write(
            f"{sha256(path)}  "
            f"{relative}\n"
        )


(
    OUT
    / ".complete"
).write_text(
    "PASS\n",
    encoding="utf-8",
)


print("=" * 78)
print(
    "FINAL EVALUATION PACKAGE V1: PASS"
)
print("=" * 78)

print(
    "output:",
    OUT,
)

print()
print(
    "===== FIRST -> CHOSEN ====="
)

print(
    improvement_df.to_string(
        index=False
    )
)

print()
print(
    "===== NUMERIC ORTHOGRAPHY AUDIT ====="
)

print(
    numeric_df.to_string(
        index=False
    )
)

print()
print(
    "===== MAIN RESULTS ====="
)

print(
    main_df.to_string(
        index=False
    )
)
