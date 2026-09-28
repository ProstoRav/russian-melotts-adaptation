#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT = Path("${PROJECT_ROOT}")

INPUT = PROJECT / "artifacts/eval/speaker_identity_eval_v1"
OUT = PROJECT / "artifacts/eval/speaker_identity_publication_plots_v1"

OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    INPUT / "yocpt_natural_reference_scores.csv",
    INPUT / "yocpt_synthetic_scores.csv",
    INPUT / "dialogs_defense_voice_consistency.csv",
    INPUT / "dialogs_actor_affinity_scores.csv",
    INPUT / ".complete",
]

for path in REQUIRED:
    if not path.is_file():
        raise RuntimeError(f"missing input: {path}")


C = {
    "reference_fill": "#D9DDE3",
    "reference_edge": "#5C6670",
    "main_fill": "#DCEAF7",
    "main_edge": "#35698F",
    "chosen_fill": "#FFE3A1",
    "chosen_edge": "#9A6200",
    "median": "#173B57",
    "trend": "#008C95",
    "whisker": "#98A3AF",
    "grid": "#D9DEE5",
    "zero": "#68737E",
    "text": "#5F6872",
    "neutral": "#35698F",
    "positive": "#B44B79",
}

plt.rcParams.update({
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
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#69747F")
    ax.spines["bottom"].set_color("#69747F")
    ax.grid(
        axis="y",
        linewidth=0.6,
        color=C["grid"],
        alpha=0.85,
    )
    ax.set_axisbelow(True)


def export(fig, stem: str):
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


def draw_boxes(
    ax,
    groups,
    positions,
    chosen_index=None,
    reference_index=None,
):
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.58,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        medianprops={
            "color": C["median"],
            "linewidth": 2.2,
        },
        whiskerprops={
            "color": C["whisker"],
            "linewidth": 0.9,
        },
        capprops={
            "color": C["whisker"],
            "linewidth": 0.9,
        },
    )

    for i, box in enumerate(bp["boxes"]):
        if (
            reference_index is not None
            and i == reference_index
        ):
            box.set_facecolor(C["reference_fill"])
            box.set_edgecolor(C["reference_edge"])
            box.set_linewidth(1.2)

        elif (
            chosen_index is not None
            and i == chosen_index
        ):
            box.set_facecolor(C["chosen_fill"])
            box.set_edgecolor(C["chosen_edge"])
            box.set_linewidth(2.0)

        else:
            box.set_facecolor(C["main_fill"])
            box.set_edgecolor(C["main_edge"])
            box.set_linewidth(1.2)

    return bp


def footer(
    fig,
    line1: str,
    line2: str | None = None,
):
    fig.text(
        0.105,
        0.038,
        line1,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=C["text"],
    )

    if line2:
        fig.text(
            0.105,
            0.019,
            line2,
            ha="left",
            va="bottom",
            fontsize=7.8,
            color=C["text"],
        )


yref = pd.read_csv(
    INPUT / "yocpt_natural_reference_scores.csv"
)

ysyn = pd.read_csv(
    INPUT / "yocpt_synthetic_scores.csv"
)

dcons = pd.read_csv(
    INPUT / "dialogs_defense_voice_consistency.csv"
)

daff = pd.read_csv(
    INPUT / "dialogs_actor_affinity_scores.csv"
)


# ------------------------------------------------------------------
# 1. YO-CPT -> spk_298 enrollment centroid
# ------------------------------------------------------------------

epochs = [1, 10, 20, 27, 30]
epoch_positions = [1, 2, 3, 4, 5]
reference_position = -0.2

reference_values = (
    yref[
        "similarity_to_enrollment_centroid"
    ].to_numpy()
)

epoch_groups = [
    ysyn.loc[
        ysyn["epoch"] == epoch,
        "similarity_to_enrollment_centroid",
    ].to_numpy()
    for epoch in epochs
]

groups = [
    reference_values,
    *epoch_groups,
]

positions = [
    reference_position,
    *epoch_positions,
]

medians = [
    float(np.median(group))
    for group in epoch_groups
]

fig, ax = plt.subplots(
    figsize=(9.2, 5.8)
)

draw_boxes(
    ax,
    groups,
    positions,
    chosen_index=4,
    reference_index=0,
)

ax.plot(
    epoch_positions,
    medians,
    color=C["trend"],
    marker="o",
    linewidth=2.3,
    markersize=5.2,
    zorder=6,
)

ax.set_xticks(positions)

ax.set_xticklabels([
    "Natural reference\n(естественная речь)",
    "E1",
    "E10",
    "E20",
    "E27\nchosen candidate",
    "E30",
])

ax.set_ylabel(
    "Cosine similarity (косинусное сходство)"
)

ax.set_title(
    "YO-CPT — Speaker Similarity (сходство с целевым голосом)\n"
    "относительно enrollment centroid spk_298",
    loc="left",
    fontweight="semibold",
    pad=12,
)

style_axes(ax)

trend_handle = Line2D(
    [0],
    [0],
    color=C["trend"],
    marker="o",
    linewidth=2.3,
    label="Median trend (динамика медианы)",
)

ax.legend(
    handles=[trend_handle],
    frameon=False,
    loc="lower right",
)

all_p10 = [
    np.percentile(group, 10)
    for group in groups
]

all_p90 = [
    np.percentile(group, 90)
    for group in groups
]

ax.set_ylim(
    min(all_p10) - 0.004,
    max(all_p90) + 0.003,
)

footer(
    fig,
    "Boxplot: box = Q1–Q3; center line = Median (медиана); whiskers = P10–P90.",
    "Natural reference не соединяется с training trajectory; E27 — выбранный checkpoint.",
)

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    top=0.86,
    bottom=0.205,
)

export(
    fig,
    "yocpt_enrollment_centroid_similarity",
)


# ------------------------------------------------------------------
# 2. YO-CPT -> matched natural reference
# ------------------------------------------------------------------

groups = [
    ysyn.loc[
        ysyn["epoch"] == epoch,
        "similarity_to_matched_natural",
    ].to_numpy()
    for epoch in epochs
]

medians = [
    float(np.median(group))
    for group in groups
]

fig, ax = plt.subplots(
    figsize=(8.7, 5.8)
)

draw_boxes(
    ax,
    groups,
    epoch_positions,
    chosen_index=3,
)

ax.plot(
    epoch_positions,
    medians,
    color=C["trend"],
    marker="o",
    linewidth=2.3,
    markersize=5.2,
    zorder=6,
)

ax.set_xticks(epoch_positions)

ax.set_xticklabels([
    "E1",
    "E10",
    "E20",
    "E27\nchosen candidate",
    "E30",
])

ax.set_ylabel(
    "Cosine similarity (косинусное сходство)"
)

ax.set_title(
    "YO-CPT — Matched-reference Speaker Similarity\n"
    "(сходство с соответствующей естественной записью)",
    loc="left",
    fontweight="semibold",
    pad=12,
)

style_axes(ax)

ax.legend(
    handles=[trend_handle],
    frameon=False,
    loc="lower right",
)

all_p10 = [
    np.percentile(group, 10)
    for group in groups
]

all_p90 = [
    np.percentile(group, 90)
    for group in groups
]

ax.set_ylim(
    min(all_p10) - 0.008,
    max(all_p90) + 0.004,
)

footer(
    fig,
    "Boxplot: box = Q1–Q3; center line = Median (медиана); whiskers = P10–P90.",
    "Каждая точка распределения — synthetic utterance vs его matched natural reference.",
)

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    top=0.86,
    bottom=0.205,
)

export(
    fig,
    "yocpt_matched_reference_similarity",
)


# ------------------------------------------------------------------
# 3. Dialogs controlled SID0 vs SID1 voice consistency
# ------------------------------------------------------------------

dialogs_epochs = [
    1,
    10,
    20,
    30,
    38,
]

dialogs_positions = [
    1,
    2,
    3,
    4,
    5,
]

groups = [
    dcons.loc[
        dcons["epoch"] == epoch,
        "sid0_sid1_cosine",
    ].to_numpy()
    for epoch in dialogs_epochs
]

medians = [
    float(np.median(group))
    for group in groups
]

fig, ax = plt.subplots(
    figsize=(8.7, 5.8)
)

draw_boxes(
    ax,
    groups,
    dialogs_positions,
    chosen_index=4,
)

ax.plot(
    dialogs_positions,
    medians,
    color=C["trend"],
    marker="o",
    linewidth=2.3,
    markersize=5.2,
    zorder=6,
)

ax.set_xticks(
    dialogs_positions
)

ax.set_xticklabels([
    "E1",
    "E10",
    "E20",
    "E30",
    "E38\nchosen candidate",
])

ax.set_ylabel(
    "Cosine similarity (косинусное сходство)"
)

ax.set_title(
    "Dialogs — Voice Identity Consistency\n"
    "(согласованность идентичности голоса SID0 ↔ SID1)",
    loc="left",
    fontweight="semibold",
    pad=12,
)

style_axes(ax)

ax.legend(
    handles=[trend_handle],
    frameon=False,
    loc="lower right",
)

all_p10 = [
    np.percentile(group, 10)
    for group in groups
]

all_p90 = [
    np.percentile(group, 90)
    for group in groups
]

ax.set_ylim(
    min(all_p10) - 0.003,
    min(
        1.0,
        max(all_p90) + 0.002,
    ),
)

footer(
    fig,
    "Boxplot: box = Q1–Q3; center line = Median (медиана); whiskers = P10–P90.",
    "Каждая пара: один текст + один seed, SID0 neutral vs SID1 positive; n = 9 фраз на эпоху.",
)

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    top=0.86,
    bottom=0.205,
)

export(
    fig,
    "dialogs_sid0_sid1_voice_consistency",
)


# ------------------------------------------------------------------
# 4. Appendix — Dialogs source actor affinity margin
# ------------------------------------------------------------------

neutral_label = (
    "dialogs_female_neutral"
)

positive_label = (
    "dialogs_female_positive"
)

actor_epochs = [
    1,
    10,
    20,
    30,
    38,
    40,
]

neutral_groups = [
    daff.loc[
        (
            daff["epoch"] == epoch
        )
        &
        (
            daff["conditioning_label"]
            == neutral_label
        ),
        "source_actor_affinity_margin",
    ].to_numpy()
    for epoch in actor_epochs
]

positive_groups = [
    daff.loc[
        (
            daff["epoch"] == epoch
        )
        &
        (
            daff["conditioning_label"]
            == positive_label
        ),
        "source_actor_affinity_margin",
    ].to_numpy()
    for epoch in actor_epochs
]

centers = np.arange(
    1,
    len(actor_epochs) + 1,
    dtype=float,
)

neutral_positions = (
    centers - 0.17
)

positive_positions = (
    centers + 0.17
)

fig, ax = plt.subplots(
    figsize=(9.2, 5.8)
)

bp_neutral = ax.boxplot(
    neutral_groups,
    positions=neutral_positions,
    widths=0.28,
    whis=(10, 90),
    showfliers=False,
    patch_artist=True,
    medianprops={
        "color": C["median"],
        "linewidth": 1.8,
    },
    whiskerprops={
        "color": C["whisker"],
        "linewidth": 0.8,
    },
    capprops={
        "color": C["whisker"],
        "linewidth": 0.8,
    },
)

bp_positive = ax.boxplot(
    positive_groups,
    positions=positive_positions,
    widths=0.28,
    whis=(10, 90),
    showfliers=False,
    patch_artist=True,
    medianprops={
        "color": C["median"],
        "linewidth": 1.8,
    },
    whiskerprops={
        "color": C["whisker"],
        "linewidth": 0.8,
    },
    capprops={
        "color": C["whisker"],
        "linewidth": 0.8,
    },
)

for box in bp_neutral["boxes"]:
    box.set_facecolor("#DCEAF7")
    box.set_edgecolor(C["neutral"])
    box.set_linewidth(1.1)

for box in bp_positive["boxes"]:
    box.set_facecolor("#F4DCE8")
    box.set_edgecolor(C["positive"])
    box.set_linewidth(1.1)

neutral_medians = [
    float(np.median(group))
    for group in neutral_groups
]

positive_medians = [
    float(np.median(group))
    for group in positive_groups
]

ax.plot(
    neutral_positions,
    neutral_medians,
    color=C["neutral"],
    marker="o",
    linewidth=1.9,
    markersize=4.5,
)

ax.plot(
    positive_positions,
    positive_medians,
    color=C["positive"],
    marker="o",
    linewidth=1.9,
    markersize=4.5,
)

ax.axhline(
    0,
    color=C["zero"],
    linestyle="--",
    linewidth=1.0,
    alpha=0.85,
)

ax.set_xticks(
    centers
)

ax.set_xticklabels([
    "E1",
    "E10",
    "E20",
    "E30",
    "E38\nchosen candidate",
    "E40",
])

ax.set_ylabel(
    "Correct − wrong actor similarity (margin)"
)

ax.set_title(
    "Dialogs — Source Actor Affinity Margin\n"
    "(близость к реальному M/S actor; appendix diagnostic)",
    loc="left",
    fontweight="semibold",
    pad=12,
)

style_axes(ax)

actor_handles = [
    Line2D(
        [0],
        [0],
        color=C["neutral"],
        marker="o",
        linewidth=1.9,
        label="Neutral conditioning",
    ),

    Line2D(
        [0],
        [0],
        color=C["positive"],
        marker="o",
        linewidth=1.9,
        label="Positive conditioning",
    ),
]

ax.legend(
    handles=actor_handles,
    frameon=False,
    loc="upper right",
    ncol=2,
)

all_values = np.concatenate(
    neutral_groups
    + positive_groups
)

p10, p90 = np.percentile(
    all_values,
    [
        10,
        90,
    ],
)

pad = max(
    0.004,
    (p90 - p10) * 0.12,
)

ax.set_ylim(
    p10 - pad,
    p90 + pad,
)

footer(
    fig,
    "Margin > 0 означает, что synthetic embedding ближе к centroid исходного actor, чем к другому.",
    "Это actor-affinity diagnostic, а не speaker-preservation metric: модель не обучалась с M/S identity labels.",
)

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    top=0.86,
    bottom=0.205,
)

export(
    fig,
    "dialogs_actor_affinity_margin_appendix",
)


# ------------------------------------------------------------------
# Compact numerical summary + provenance
# ------------------------------------------------------------------

plot_summary = {
    "yocpt_natural_median":
        float(
            yref[
                "similarity_to_enrollment_centroid"
            ].median()
        ),

    "yocpt_e27_centroid_median":
        float(
            ysyn.loc[
                ysyn["epoch"] == 27,
                "similarity_to_enrollment_centroid",
            ].median()
        ),

    "yocpt_e30_centroid_median":
        float(
            ysyn.loc[
                ysyn["epoch"] == 30,
                "similarity_to_enrollment_centroid",
            ].median()
        ),

    "yocpt_e27_matched_median":
        float(
            ysyn.loc[
                ysyn["epoch"] == 27,
                "similarity_to_matched_natural",
            ].median()
        ),

    "dialogs_e38_sid_cos_median":
        float(
            dcons.loc[
                dcons["epoch"] == 38,
                "sid0_sid1_cosine",
            ].median()
        ),
}

(
    OUT
    / "plot_summary.json"
).write_text(
    json.dumps(
        plot_summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


provenance = {
    "version": 1,

    "input": str(
        INPUT
    ),

    "presentation_boxplot": {
        "box": "Q1-Q3",
        "center": "median",
        "whiskers": "P10-P90",
        "outliers_displayed": False,
    },

    "chosen_candidates": {
        "yocpt": 27,
        "dialogs": 38,
    },

    "note": (
        "E30/E40 are shown only as epochs; "
        "no post-winner terminology is used."
    ),
}

(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        provenance,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


checksum_targets = sorted(
    path
    for path in OUT.iterdir()
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
        f.write(
            f"{sha256(path)}  "
            f"{path.name}\n"
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
    "SPEAKER IDENTITY PUBLICATION PLOTS V1: PASS"
)
print("=" * 78)

print(
    "input:",
    INPUT,
)

print(
    "output:",
    OUT,
)

print(
    "files:",
    len(
        checksum_targets
    ),
)

print()
print(
    json.dumps(
        plot_summary,
        ensure_ascii=False,
        indent=2,
    )
)
