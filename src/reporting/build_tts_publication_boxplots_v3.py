#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT = Path("${PROJECT_ROOT}")

INPUT = (
    PROJECT
    / "artifacts/eval/"
      "tts_trajectory_metrics_v2"
)

OUT = (
    PROJECT
    / "artifacts/eval/"
      "tts_publication_boxplots_v3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PER_AUDIO = pd.read_csv(
    INPUT / "per_audio_metrics.csv"
)

AGG = pd.read_csv(
    INPUT / "aggregate_overall.csv"
)


C = {
    "reference_fill": "#D9DDE3",
    "reference_edge": "#5C6670",

    "main_fill": "#DCEAF7",
    "main_edge": "#35698F",

    "chosen_fill": "#FFE3A1",
    "chosen_edge": "#9A6200",

    "median": "#173B57",
    "trend": "#008C95",
    "aggregate": "#C02F7D",

    "whisker": "#98A3AF",
    "grid": "#D9DEE5",
    "secondary_text": "#5F6872",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


EXPERIMENTS = {
    "dialogs": {
        "title": "Dialogs",
        "epochs": [
            1,
            10,
            20,
            30,
            38,
            40,
        ],
        "chosen": 38,
    },

    "yocpt": {
        "title": "YO-CPT",
        "epochs": [
            1,
            10,
            20,
            27,
            30,
        ],
        "chosen": 27,
    },
}


def values_for(
    experiment,
    metric,
    epoch=None,
    reference=False,
):
    x = PER_AUDIO[
        PER_AUDIO[
            "experiment"
        ]
        == experiment
    ]

    if reference:
        x = x[
            x[
                "checkpoint_role"
            ]
            == "REFERENCE"
        ]
    else:
        x = x[
            x[
                "epoch"
            ]
            == epoch
        ]

    values = (
        x[metric]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if not len(values):
        raise RuntimeError(
            f"No values: "
            f"{experiment} "
            f"{metric} "
            f"epoch={epoch} "
            f"reference={reference}"
        )

    return values


def aggregate_value(
    experiment,
    column,
    epoch=None,
    reference=False,
):
    x = AGG[
        AGG[
            "experiment"
        ]
        == experiment
    ]

    if reference:
        x = x[
            x[
                "checkpoint_role"
            ]
            == "REFERENCE"
        ]
    else:
        x = x[
            x[
                "epoch"
            ]
            == epoch
        ]

    if len(x) != 1:
        raise RuntimeError(
            f"Aggregate lookup failed: "
            f"{experiment} "
            f"{column} "
            f"{epoch}"
        )

    return float(
        x.iloc[0][column]
    )


def export(
    fig,
    stem,
):
    fig.savefig(
        OUT / f"{stem}.svg",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def build_plot(
    experiment,
    metric,
    aggregate_column,
    title_metric,
    ylabel,
    stem,
    percentage=False,
    include_reference=False,
    aggregate_label="Mean",
):
    cfg = EXPERIMENTS[
        experiment
    ]

    epochs = cfg[
        "epochs"
    ]

    chosen = cfg[
        "chosen"
    ]

    epoch_groups = [
        values_for(
            experiment,
            metric,
            epoch=e,
        )
        for e in epochs
    ]

    if percentage:
        epoch_groups = [
            g * 100.0
            for g in epoch_groups
        ]

    epoch_aggregates = [
        aggregate_value(
            experiment,
            aggregate_column,
            epoch=e,
        )
        * (
            100.0
            if percentage
            else 1.0
        )
        for e in epochs
    ]

    epoch_positions = list(
        range(
            1,
            len(epochs) + 1,
        )
    )

    groups = []
    positions = []
    labels = []
    roles = []
    aggregates = []

    if include_reference:
        ref_values = values_for(
            experiment,
            metric,
            reference=True,
        )

        if percentage:
            ref_values = (
                ref_values
                * 100.0
            )

        ref_aggregate = (
            aggregate_value(
                experiment,
                aggregate_column,
                reference=True,
            )
            * (
                100.0
                if percentage
                else 1.0
            )
        )

        groups.append(
            ref_values
        )

        positions.append(
            -0.2
        )

        labels.append(
            "Natural reference\n"
            "(естественная речь)"
        )

        roles.append(
            "reference"
        )

        aggregates.append(
            ref_aggregate
        )

    for epoch, xpos, group, agg in zip(
        epochs,
        epoch_positions,
        epoch_groups,
        epoch_aggregates,
    ):
        groups.append(
            group
        )

        positions.append(
            xpos
        )

        labels.append(
            f"E{epoch}"
            + (
                "\nchosen candidate"
                if epoch == chosen
                else ""
            )
        )

        roles.append(
            "chosen"
            if epoch == chosen
            else "main"
        )

        aggregates.append(
            agg
        )

    fig, ax = plt.subplots(
        figsize=(
            9.2,
            5.8,
        )
    )

    #
    # Presentation-focused boxplot:
    # Q1-Q3 + median stay standard.
    # Whiskers show P10-P90 instead of 1.5*IQR
    # to keep long tails from dominating the
    # visible scale.
    #
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.58,
        whis=(
            10,
            90,
        ),
        showfliers=False,
        patch_artist=True,

        medianprops={
            "color": C[
                "median"
            ],
            "linewidth": 2.2,
        },

        whiskerprops={
            "color": C[
                "whisker"
            ],
            "linewidth": 0.9,
        },

        capprops={
            "color": C[
                "whisker"
            ],
            "linewidth": 0.9,
        },
    )

    for box, role in zip(
        bp["boxes"],
        roles,
    ):
        if role == "reference":
            box.set_facecolor(
                C[
                    "reference_fill"
                ]
            )

            box.set_edgecolor(
                C[
                    "reference_edge"
                ]
            )

            box.set_linewidth(
                1.2
            )

        elif role == "chosen":
            box.set_facecolor(
                C[
                    "chosen_fill"
                ]
            )

            box.set_edgecolor(
                C[
                    "chosen_edge"
                ]
            )

            box.set_linewidth(
                2.0
            )

        else:
            box.set_facecolor(
                C[
                    "main_fill"
                ]
            )

            box.set_edgecolor(
                C[
                    "main_edge"
                ]
            )

            box.set_linewidth(
                1.2
            )

    #
    # Trend is defined only across actual
    # training checkpoints. Natural reference
    # is intentionally not connected.
    #
    medians = [
        float(
            np.median(
                g
            )
        )
        for g in epoch_groups
    ]

    ax.plot(
        epoch_positions,
        medians,
        color=C[
            "trend"
        ],
        marker="o",
        linewidth=2.3,
        markersize=5.2,
        zorder=6,
    )

    #
    # Report-level aggregate:
    # corpus CER/WER or mean MAPE.
    #
    ax.scatter(
        positions,
        aggregates,
        marker="D",
        s=44,
        color=C[
            "aggregate"
        ],
        edgecolors="white",
        linewidths=0.9,
        zorder=7,
    )

    #
    # Focus vertical scale on P90 and report
    # aggregates. Values above P90 remain in
    # all calculations but are omitted from
    # this presentation view.
    #
    p90_values = [
        float(
            np.percentile(
                g,
                90,
            )
        )
        for g in groups
    ]

    ceiling = (
        max(
            max(
                p90_values
            ),
            max(
                aggregates
            ),
        )
        * 1.12
    )

    if ceiling <= 0:
        ceiling = 1.0

    ax.set_ylim(
        0,
        ceiling,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        (
            f"{cfg['title']} — "
            f"{title_metric} "
            "по эпохам"
        ),
        loc="left",
        fontweight="semibold",
        pad=12,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_color(
        "#69747F"
    )

    ax.spines[
        "bottom"
    ].set_color(
        "#69747F"
    )

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.6,
        color=C[
            "grid"
        ],
        alpha=0.85,
    )

    ax.set_axisbelow(
        True
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=C[
                "trend"
            ],
            marker="o",
            linewidth=2.3,
            label=(
                "Median trend "
                "(динамика медианы)"
            ),
        ),

        Line2D(
            [0],
            [0],
            color="none",
            marker="D",
            markerfacecolor=C[
                "aggregate"
            ],
            markeredgecolor="white",
            markersize=7,
            label=aggregate_label,
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper right",
        ncol=2,
        handlelength=2.1,
        columnspacing=1.6,
    )

    #
    # Two intentionally short lines so caption
    # never makes the exported image wider than
    # the chart itself.
    #
    fig.text(
        0.105,
        0.037,
        (
            "Boxplot: box = Q1–Q3; center line = "
            "Median (медиана); whiskers = P10–P90."
        ),
        ha="left",
        va="bottom",
        fontsize=7.9,
        color=C[
            "secondary_text"
        ],
    )

    fig.text(
        0.105,
        0.018,
        (
            "Значения за пределами P10–P90 не показаны "
            "на графике, но полностью учитываются "
            "в расчётах."
        ),
        ha="left",
        va="bottom",
        fontsize=7.9,
        color=C[
            "secondary_text"
        ],
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.90,
        bottom=0.205,
    )

    export(
        fig,
        (
            f"{experiment}_"
            f"{stem}"
        ),
    )


# ============================================================
# WER
# ============================================================

for experiment in (
    "dialogs",
    "yocpt",
):
    build_plot(
        experiment=experiment,
        metric="wer",
        aggregate_column="corpus_wer",
        title_metric=(
            "WER (ошибка по словам)"
        ),
        ylabel=(
            "WER (ошибка по словам), %"
        ),
        stem="wer_boxplot",
        percentage=True,
        include_reference=True,
        aggregate_label="Corpus WER",
    )


# ============================================================
# CER
# ============================================================

for experiment in (
    "dialogs",
    "yocpt",
):
    build_plot(
        experiment=experiment,
        metric="cer",
        aggregate_column="corpus_cer",
        title_metric=(
            "CER (ошибка по символам)"
        ),
        ylabel=(
            "CER (ошибка по символам), %"
        ),
        stem="cer_boxplot",
        percentage=True,
        include_reference=True,
        aggregate_label="Corpus CER",
    )


# ============================================================
# Duration MAPE
# ============================================================

for experiment in (
    "dialogs",
    "yocpt",
):
    build_plot(
        experiment=experiment,
        metric=(
            "duration_abs_pct_error"
        ),
        aggregate_column=(
            "duration_mape_mean"
        ),
        title_metric=(
            "Duration MAPE "
            "(ошибка длительности)"
        ),
        ylabel=(
            "Duration MAPE "
            "(ошибка длительности), %"
        ),
        stem="duration_mape_boxplot",
        percentage=False,
        include_reference=False,
        aggregate_label="Mean MAPE",
    )


# ============================================================
# F0 MAPE
# ============================================================

for experiment in (
    "dialogs",
    "yocpt",
):
    build_plot(
        experiment=experiment,
        metric=(
            "f0_median_abs_pct_error"
        ),
        aggregate_column=(
            "f0_mape_mean"
        ),
        title_metric=(
            "F0 MAPE "
            "(ошибка медианы F0)"
        ),
        ylabel=(
            "F0 MAPE "
            "(ошибка медианы F0), %"
        ),
        stem="f0_mape_boxplot",
        percentage=False,
        include_reference=False,
        aggregate_label="Mean F0 MAPE",
    )


manifest = {
    "version": 3,

    "input": str(
        INPUT
    ),

    "purpose": (
        "presentation-focused publication plots"
    ),

    "distribution_view": {
        "box": "Q1-Q3",
        "center": "median",
        "whiskers": "P10-P90",
        "fliers": (
            "not displayed; retained in calculations"
        ),
    },

    "trend": (
        "median across every measured checkpoint"
    ),

    "chosen_candidate": {
        "dialogs": "E38",
        "yocpt": "E27",
    },

    "special_label_for_later_epoch": False,

    "formats": [
        "svg",
        "pdf",
        "png",
    ],
}


(
    OUT / "manifest.json"
).write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


(
    OUT / ".complete"
).write_text(
    "PASS\n",
    encoding="utf-8",
)


print("=" * 78)
print(
    "TTS PUBLICATION BOXPLOTS V3: PASS"
)
print("=" * 78)

for path in sorted(
    OUT.glob(
        "*.svg"
    )
):
    print(
        path.name
    )

print()
print(
    "output:",
    OUT,
)
