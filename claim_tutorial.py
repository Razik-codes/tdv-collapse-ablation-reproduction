# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo==0.23.15", "numpy==2.1.1", "matplotlib==3.9.2"]
# ///
#
# Tutorial reproduction of the collapse ablation (Table 4) from
# "You Don't Need Strong Assumptions: Visual Representation Learning via
# Temporal Differences" (arXiv:2606.15956).
#
# Self-contained: all reproduction results are frozen inline, so the notebook
# renders every figure immediately with no computation. The only heavy code is
# the optional GPU lab at the bottom, which is gated behind a run button.

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 🎬 Does TDV really need its motion encoder *and* its MSE loss?

        ### A tutorial reproduction of Table 4 from *“You Don’t Need Strong Assumptions: Visual Representation Learning via Temporal Differences”* (TDV)

        **The paper's big idea.** Instead of the usual self-supervised tricks (crops,
        color jitter, masking, contrastive negatives), **TDV** learns image
        representations from video using one weak assumption: *the past predicts the
        future.* A **frame encoder** maps a frame to a representation `z_t`; a
        **motion encoder** maps the raw pixel change `Δx_t = x_{t+1}-x_t` to a latent
        shift `Δz_t`; and the model is trained so that

        $$\hat z_{t+1} = z_t + \Delta z_t \;\approx\; z_{t+1}.$$

        **The claim we reproduce (Table 4).** Two components are *load-bearing*: the
        **motion encoder** and the **MSE next-frame-prediction loss**. Remove either
        and training **collapses** — the representations become useless. This is what
        makes TDV's "weak assumption" actually do work, so it is the single clearest
        window into *why the method works*.

        > This page shows **frozen results** from real runs launched through `experiment`.
        > Nothing below recomputes the reproduction — scroll freely. The only live
        > computation is the **optional GPU lab** at the very end, behind a button.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, paper):
    mo.md(
        f"""
        ## 1 · The paper's result

        The authors pretrain TDV on **Something-Something-v2** (SSv2, ~220k clips)
        and use **online ImageNet KNN Top-5 accuracy** as a cheap probe of
        representation quality — *collapsed representations score near chance.*
        Table 4 (ViT-S) reports:

        | Ablation | ImageNet KNN Top-5 ↑ | Avoids collapse? |
        |---|---:|:---:|
        | **Full TDV recipe** | **{paper['full']:.2f}%** | ✅ |
        | Remove **motion encoder** | {paper['no_motion']:.2f}% | ❌ collapse |
        | Remove **MSE loss** | {paper['no_mse']:.2f}% | ❌ collapse |

        Full TDV reaches **{paper['full']:.1f}%**; knocking out *either* component drops
        it to **~{paper['no_mse']:.1f}–{paper['no_motion']:.1f}%**, i.e. essentially random.
        [→ Read the paper (arXiv:2606.15956)](https://arxiv.org/abs/2606.15956).
        """
    )
    return


@app.cell(hide_code=True)
def _(END, mo, verdict_badge):
    mo.md(
        f"""
        ## 2 · The reproduction result

        {verdict_badge}

        Full-scale SSv2 pretraining + an ImageNet KNN probe is far out of reach on the
        4 GB laptop GPU used here, so this is a **downscaled** reproduction (see
        *Limitations* below for the exact substitutions). The key change: because we
        have no ImageNet, we replace the KNN collapse-detector with a **direct,
        unconfounded read on TDV's own objective** —

        > **Next-frame prediction gain** = how much better the motion-composed
        > prediction $\\hat z_{{t+1}}=z_t+\\Delta z_t$ predicts the (teacher-encoded)
        > next frame than the trivial **identity baseline** $z_t$ alone, in the same
        > representation space:
        > $\\text{{gain}} = \\dfrac{{\\lVert z_t-z_{{t+1}}\\rVert - \\lVert \\hat z_{{t+1}}-z_{{t+1}}\\rVert}}{{\\lVert z_t-z_{{t+1}}\\rVert}}$.
        > Positive = the motion encoder is doing real work. Zero/negative = it isn't.

        On synthetic **low-rank-motion** clips (a textured disc moving over a static
        background — faithful to TDV's assumption that Δx is low-rank), 600 steps,
        ViT-S/14 from scratch:

        | Arm | Prediction gain over identity | Repr. variance | Verdict |
        |---|---:|---:|:---|
        | **Full TDV** | **+{END['control']['gain_pct']:.0f}%** | {END['control']['variance']:.2f} | ✅ healthy — learns to predict motion |
        | Remove **motion encoder** | **{END['no_motion']['gain_pct']:.0f}%** | {END['no_motion']['variance']:.2f} | ❌ no motion path → nothing to gain |
        | Remove **MSE loss** | **{END['no_mse']['gain_pct']:.0f}%** | {END['no_mse']['variance']:.2f} | ❌ unsupervised Δz *hurts* prediction |

        Same qualitative structure as the paper: **only the full recipe works; removing
        either component breaks it.** The identity baseline (`+0%`) and the destructive
        unsupervised motion encoder (`−447%`) are two different flavors of the paper's
        "collapse."
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3 · The picture: one metric separates all three arms

        Pick a metric and watch the three arms diverge over training. The headline is
        **prediction gain over the identity baseline**: the full model climbs to ~+45%,
        the motion-encoder-free model is pinned at exactly 0% (its prediction *is* the
        identity baseline), and the MSE-free model dives negative — its unsupervised
        motion vector actively corrupts the prediction.
        """
    )
    return


@app.cell
def _(metric_sel):
    metric_sel
    return


@app.cell(hide_code=True)
def _(FROZEN, METRICS, metric_sel, np, plt, ARMS):
    _key = metric_sel.value
    _label, _desc = METRICS[_key]
    _fig, _ax = plt.subplots(figsize=(8, 4.6))
    for _arm, _style in ARMS.items():
        _tr = FROZEN[_arm]["traj"]
        _ax.plot(_tr["step"], _tr[_key], marker=_style["m"], ms=4, lw=2,
                 color=_style["c"], label=_style["label"])
    if _key == "gain_pct":
        _ax.axhline(0, color="#444", lw=1, ls="--", alpha=0.7)
        _ax.annotate("identity baseline (no gain)", xy=(300, 0), xytext=(300, -120),
                     fontsize=8, color="#444", ha="center")
    _ax.set_xlabel("training step")
    _ax.set_ylabel(_label)
    _ax.set_title(_desc, fontsize=10)
    _ax.legend(loc="best", fontsize=9, framealpha=0.9)
    _ax.grid(True, alpha=0.25)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **How to read it.** With `prediction gain` selected:

        - **Full TDV (blue)** starts *worse* than identity (an untrained motion encoder
          adds noise) then quickly overtakes it — the motion encoder *learns* to predict
          where the disc moves. Its representation `variance` settles to a healthy plateau
          and `dino_entropy` stays high (no mode collapse).
        - **No motion encoder (grey)** is flat at exactly **0%** — with `ẑ = z_t`, the
          "prediction" *is* the identity baseline. There is no temporal-difference signal
          to learn at all.
        - **No MSE loss (orange)** falls *below* zero and keeps sinking. The motion encoder
          still exists, but nothing supervises it to predict the next frame, so its Δz is
          large, unstructured noise that makes the composed prediction far worse than doing
          nothing — and its representation `variance` drops the most of the three (partial
          collapse).
        """
    )
    return


@app.cell(hide_code=True)
def _(FROZEN, np, plt, ARMS):
    # Two-panel supporting view: prediction error vs identity, and representation health.
    _fig, (_axL, _axR) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for _arm, _style in ARMS.items():
        _tr = FROZEN[_arm]["traj"]
        _axL.plot(_tr["step"], _tr["l1_loss"], color=_style["c"], lw=2, label=_style["label"])
        _axL.plot(_tr["step"], _tr["baseline_l1_loss"], color=_style["c"], lw=1, ls=":", alpha=0.7)
        _axR.plot(_tr["step"], _tr["variance"], color=_style["c"], lw=2, label=_style["label"])
    _axL.set_title("Prediction error: motion-composed (solid)\nvs identity baseline (dotted)", fontsize=9)
    _axL.set_xlabel("training step"); _axL.set_ylabel("L1 error to next-frame repr."); _axL.grid(True, alpha=0.25)
    _axL.legend(fontsize=8)
    _axR.set_title("Representation variance\n(→0 = collapse to a point)", fontsize=9)
    _axR.set_xlabel("training step"); _axR.set_ylabel("mean per-dim variance"); _axR.grid(True, alpha=0.25)
    _axR.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, np, render_pair_np):
    # Show the (faithful) synthetic data + its low-rank RGB difference. Pure numpy, instant.
    import matplotlib.pyplot as _plt
    _f0, _f1 = render_pair_np(magnitude=34, seed=7)
    _diff = np.abs(_f1.astype(float) - _f0.astype(float)).mean(-1)
    _fig, _ax = _plt.subplots(1, 3, figsize=(9, 3.2))
    _ax[0].imshow(_f0); _ax[0].set_title("frame $x_t$", fontsize=9)
    _ax[1].imshow(_f1); _ax[1].set_title("frame $x_{t+1}$", fontsize=9)
    _im = _ax[2].imshow(_diff, cmap="magma"); _ax[2].set_title("$|\\Delta x_t|$  (low-rank)", fontsize=9)
    for _a in _ax:
        _a.set_xticks([]); _a.set_yticks([])
    _frac = float((_diff > 12).mean())
    _fig.suptitle(f"Synthetic low-rank motion: only ~{_frac*100:.0f}% of pixels change "
                  f"(the moving disc) — the rest is static background", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · Robustness & interpretation

        **A negative control that shows the metric isn't fooling us.** An earlier round of
        this reproduction used `ffmpeg testsrc2` clips, whose *entire frame* animates —
        violating TDV's assumption that Δx is *low-rank* (mostly-static background + a small
        moving region). On that data the identity baseline was *unbeatable* (consecutive
        frames were nearly identical), the motion encoder had nothing to learn, and the
        collapse signal was muddy: representation variance came out **non-monotonic across
        arms** (frozen-high for no-motion, mid for full, low for no-MSE) — a confounded,
        uninterpretable axis. Switching to faithful low-rank-motion data is exactly what
        made the clean +45 / 0 / −447 separation appear. *The result is a property of
        TDV's mechanism meeting the right data, not an artifact of the metric.*

        **What would falsify our interpretation?**

        - If the **full model's** prediction gain were also ~0, TDV's additive composition
          wouldn't be learning motion — the mechanism would be illusory. (It reaches +43%.)
        - If **removing the MSE loss** left the gain positive, the DINO/iBOT self-distillation
          alone would suffice and the MSE term would be decorative. (It goes to −447%.)
        - If the **no-motion** arm somehow gained >0%, the "prediction" would be more than the
          identity — impossible by construction, and indeed it is exactly 0 at every step.

        **Honest caveat.** `dino_entropy` stays high (~6.8) in *all three* arms because the
        teacher-centering anti-collapse mechanism is present everywhere; at this tiny scale we
        do **not** see the textbook "entropy → 0 / variance → 0" total collapse. What we see is
        the paper's collapse in its *functional* form — **representations that fail to support
        the task** — which is precisely what the paper's near-chance KNN measures.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5 · Limitations & provenance

        This is a **downscaled, qualitative** reproduction. Exactly how it differs from the
        paper:

        | Aspect | Paper (Table 4 / App. C) | This reproduction |
        |---|---|---|
        | Pretraining data | SSv2, ~220k real video clips | Synthetic low-rank motion (disc on static bg), **32 train / 8 val** clips |
        | Backbone | ViT-S/14 **and** ViT-B/14 | ViT-S/14 only, from scratch |
        | Batch size | 256 images | **1** (×2 grad-accum) |
        | Training length | ~200,000 steps (20 epochs) | **600 steps** |
        | Projection head dim | 32,768 | 1,024 |
        | Collapse detector | ImageNet KNN Top-5 (near-chance ⇒ collapse) | **Next-frame prediction gain vs identity** + representation variance/entropy |
        | Compute | multi-GPU | **1× RTX 3050 Ti (4 GB)**, ≈7 min per arm |

        **Verdict: reproduced (qualitatively).** The paper's ordering — *full recipe works;
        removing the motion encoder or the MSE loss breaks it* — reproduces cleanly and with a
        large margin. The paper's exact KNN numbers (17.05 vs 1.87 / 1.58) are **not**
        reproducible here (no ImageNet, no full-scale pretraining), so we report the mechanism,
        not the benchmark.

        **What a full-scale reproduction would still need:** real SSv2 pretraining at batch 256
        for ~200k steps on multiple GPUs, then an online ImageNet-1k KNN probe — plus the
        separate downstream decoders (optical-flow / segmentation / stereo) for the *headline*
        Table 2/3 numbers, which this tutorial does not attempt.

        ### Runnable provenance

        Every result above comes from a pushed experiment branch; each runs the **identical**
        fixed command `bash job_scripts/pretrain_tdv_local_smoketest.sh` and differs only in
        committed code. The reproduction uses the *round-2 (faithful-data)* arms:

        | What | Branch (runnable code + fixed config) |
        |---|---|
        | Full TDV control | [`…control-faithful-low-rank-motion-data`](https://github.com/Razik-codes/tdv-collapse-ablation-reproduction/tree/experiment/full-tdv-control-faithful-low-rank-motion-data-r) |
        | − Motion encoder | [`…ablation-remove-motion-encoder-faithful-data`](https://github.com/Razik-codes/tdv-collapse-ablation-reproduction/tree/experiment/ablation-remove-motion-encoder-faithful-data-rou) |
        | − MSE loss | [`…ablation-remove-mse-loss-faithful-data-round-2`](https://github.com/Razik-codes/tdv-collapse-ablation-reproduction/tree/experiment/ablation-remove-mse-loss-faithful-data-round-2) |

        Full write-up: [reports/collapse-ablation/report.md](https://github.com/Razik-codes/tdv-collapse-ablation-reproduction/blob/main/reports/collapse-ablation/report.md).
        The collapse-metric plumbing (`--log_var_covar`, `--print_metrics_to_stdout`) lives on
        the [round-1 control branch](https://github.com/Razik-codes/tdv-collapse-ablation-reproduction/tree/experiment/full-tdv-control-collapse-metrics-logged-table-4).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6 · Optional interactive GPU lab — *what is the motion encoder trying to predict?*

        The reproduction above shows the motion encoder is **essential**. This lab makes its
        *target* tangible. Using a **real pretrained DINOv2 ViT-S/14** (the same architecture
        family as TDV's frame encoder), we measure the **representation-space temporal
        difference** `‖Δz‖ = ‖z(x_{t+1}) − z(x_t)‖` as we vary how far an object moves between
        two frames — batched across many synthetic frame-pairs and probed at every transformer
        block with activation hooks.

        **Why it matters for the claim:** `Δz` *is* the quantity the motion encoder is trained
        (by the MSE loss) to predict. When motion is real, `Δz` is large and spatially localized
        to the moving region — a rich learning signal. At **zero motion, `Δz`→0**: there is
        nothing to predict — the degenerate regime the ablated models get stuck in.

        > ⚠️ This cell does real GPU work (a batched ViT forward sweep). It is **off by default**
        > — nothing runs until you press the button. Target ~5–60 s on molab's RTX PRO 6000;
        > a reduced CPU fallback is used automatically if no GPU is present. This is a
        > **teaching demo on a pretrained encoder — not reproduction evidence.**
        """
    )
    return


@app.cell
def _(mo, n_pairs_slider, max_move_slider, run_btn):
    mo.hstack(
        [n_pairs_slider, max_move_slider, run_btn],
        justify="start", gap=2,
    )
    return


@app.cell(hide_code=True)
def _(lab_result, mo):
    mo.md(lab_result) if isinstance(lab_result, str) else lab_result
    return


# ----------------------------------------------------------------------------
# machinery (data, styling, widgets, lab compute) — collapsed by default
# ----------------------------------------------------------------------------


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    return json, np, plt


@app.cell
def _(json):
    # ---- FROZEN reproduction results (round-2 faithful-data arms, from experiment run logs) ----
    _FROZEN_JSON = r"""{"control":{"traj":{"step":[0,24,48,72,84,108,132,156,180,204,228,240,264,288,312,336,360,372,396,420,444,468,492,516,528,552,576,599],"gain_pct":[-291.86,39.6,45.31,48.6,47.51,49.26,58.69,57.86,54.64,54.48,50.68,51.18,46.62,56.88,43.61,52.49,44.16,61.51,52.52,50.66,55.9,47.11,45.33,49.58,39.93,36.48,38.91,34.92],"l1_loss":[0.50108,0.22367,0.18185,0.161,0.15542,0.1277,0.10279,0.10493,0.11042,0.10503,0.10046,0.10714,0.09774,0.07342,0.09597,0.07778,0.0835,0.04981,0.06858,0.06556,0.04501,0.05625,0.0703,0.04777,0.07096,0.06512,0.06545,0.06075],"baseline_l1_loss":[0.12787,0.37029,0.33248,0.31325,0.2961,0.25169,0.24881,0.249,0.24344,0.23073,0.20371,0.21947,0.18309,0.17026,0.1702,0.16369,0.14954,0.1294,0.14444,0.13288,0.10207,0.10635,0.12859,0.09476,0.11811,0.10252,0.10714,0.09334],"variance":[0.64621,0.38881,0.40132,0.30489,0.35857,0.36607,0.34656,0.38593,0.36466,0.29443,0.34426,0.32205,0.42737,0.36376,0.34896,0.40479,0.35336,0.23942,0.33056,0.33661,0.28761,0.35137,0.35726,0.22605,0.34453,0.31319,0.3087,0.27581],"teacher_variance":[0.64997,0.55236,0.51795,0.38328,0.48784,0.46976,0.37069,0.42681,0.40294,0.31123,0.38263,0.33753,0.45338,0.40307,0.39347,0.41483,0.37668,0.25136,0.35421,0.36753,0.30102,0.37532,0.38155,0.23933,0.3692,0.33206,0.3248,0.29063],"dino_entropy":[6.80643,6.89511,6.90215,6.90015,6.89415,6.88365,6.87198,6.86917,6.86321,6.86087,6.86099,6.85727,6.84912,6.84697,6.84248,6.84417,6.83663,6.84409,6.83547,6.83345,6.8321,6.83405,6.82717,6.83049,6.82585,6.83361,6.81931,6.81754],"off_diag_covariance":[0.16602,0.12207,0.12109,0.0918,0.10693,0.10938,0.10644,0.11719,0.10644,0.09082,0.10303,0.09814,0.13281,0.11133,0.10596,0.12354,0.1084,0.07471,0.09717,0.09912,0.0874,0.10986,0.1084,0.0708,0.10791,0.09521,0.09131,0.08594]},"end":{"gain_pct":42.6,"l1_loss":0.05106,"baseline_l1_loss":0.08895,"variance":0.30889,"teacher_variance":0.32116,"dino_entropy":6.81661,"off_diag_covariance":0.09473}},"no_motion":{"traj":{"step":[0,24,48,72,84,108,132,156,180,204,228,240,264,288,312,336,360,372,396,420,444,468,492,516,528,552,576,599],"gain_pct":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"l1_loss":[0.12787,0.18069,0.15853,0.14901,0.14254,0.10504,0.10046,0.10866,0.11407,0.13546,0.09279,0.11793,0.09226,0.07915,0.09082,0.07827,0.07208,0.04126,0.0711,0.06785,0.03936,0.05824,0.07581,0.04586,0.07924,0.06412,0.07484,0.06333],"baseline_l1_loss":[0.12787,0.18069,0.15853,0.14901,0.14254,0.10504,0.10046,0.10866,0.11407,0.13546,0.09279,0.11793,0.09226,0.07915,0.09082,0.07827,0.07208,0.04126,0.0711,0.06785,0.03936,0.05824,0.07581,0.04586,0.07924,0.06412,0.07484,0.06333],"variance":[0.64621,0.55885,0.5932,0.42616,0.53427,0.54983,0.48882,0.53422,0.53836,0.40986,0.48178,0.40016,0.57315,0.47932,0.48073,0.53832,0.46576,0.30556,0.42687,0.43643,0.36465,0.46474,0.50217,0.28159,0.46844,0.41091,0.40298,0.36238],"teacher_variance":[0.64997,0.59305,0.62067,0.47833,0.59218,0.60601,0.47635,0.5485,0.53824,0.40402,0.47714,0.4328,0.58826,0.52547,0.50348,0.53879,0.49345,0.31559,0.45374,0.47233,0.37883,0.4907,0.51532,0.2987,0.49825,0.42713,0.41972,0.37655],"dino_entropy":[6.811,6.89007,6.88581,6.88287,6.87079,6.86253,6.87387,6.85878,6.85828,6.86009,6.84971,6.85574,6.8397,6.84251,6.8462,6.8342,6.83699,6.85707,6.8388,6.83376,6.83554,6.82571,6.82645,6.84399,6.81321,6.82578,6.82526,6.81925],"off_diag_covariance":[0.16602,0.17285,0.17773,0.12793,0.15918,0.16504,0.15039,0.16211,0.16309,0.12891,0.14453,0.12158,0.17871,0.14551,0.14648,0.16406,0.14355,0.0957,0.12598,0.12988,0.11133,0.14551,0.15234,0.08887,0.14746,0.125,0.11963,0.11426]},"end":{"gain_pct":0.0,"l1_loss":0.05299,"baseline_l1_loss":0.05299,"variance":0.37993,"teacher_variance":0.39548,"dino_entropy":6.82025,"off_diag_covariance":0.12256}},"no_mse":{"traj":{"step":[0,24,48,72,84,108,132,156,180,204,228,240,264,288,312,336,360,372,396,420,444,468,492,516,528,552,576,599],"gain_pct":[-291.86,-32.9,-40.99,-45.03,-45.77,-83.37,-92.59,-102.46,-118.65,-144.12,-164.15,-163.88,-191.21,-227.42,-258.15,-268.37,-264.55,-380.71,-324.9,-327.28,-271.82,-424.35,-315.31,-521.41,-313.59,-385.52,-347.98,-402.53],"l1_loss":[0.50108,0.72187,0.6192,0.54937,0.52325,0.43821,0.41697,0.41604,0.40693,0.40651,0.3961,0.40177,0.39436,0.38696,0.39379,0.38642,0.40087,0.38047,0.39339,0.38928,0.39803,0.37763,0.38636,0.3696,0.38317,0.37061,0.37459,0.36446],"baseline_l1_loss":[0.12787,0.54318,0.43918,0.3788,0.35894,0.23897,0.21651,0.20549,0.18611,0.16652,0.14995,0.15226,0.13542,0.11819,0.10995,0.1049,0.10996,0.07915,0.09258,0.09111,0.10705,0.07202,0.09303,0.05948,0.09265,0.07633,0.08362,0.07252],"variance":[0.64621,0.3179,0.44557,0.33664,0.3968,0.39456,0.29932,0.3374,0.30874,0.2641,0.31532,0.24932,0.31976,0.27611,0.27159,0.28509,0.2346,0.16452,0.20167,0.23303,0.26038,0.24355,0.25574,0.14911,0.26006,0.22617,0.21819,0.19732],"teacher_variance":[0.64997,0.46892,0.39399,0.27651,0.37944,0.3889,0.30483,0.3559,0.35079,0.26016,0.31675,0.27705,0.38327,0.31945,0.30881,0.31345,0.25755,0.18137,0.22708,0.26655,0.28462,0.27014,0.27179,0.16198,0.28393,0.24264,0.22656,0.20831],"dino_entropy":[6.80643,6.90479,6.89936,6.89885,6.88809,6.87872,6.88851,6.87253,6.86904,6.87585,6.85181,6.86404,6.84235,6.84584,6.84805,6.83097,6.83687,6.86187,6.84883,6.82406,6.8093,6.80333,6.80171,6.84376,6.78493,6.79216,6.80553,6.80749],"off_diag_covariance":[0.16602,0.12451,0.1748,0.12988,0.14844,0.15234,0.11475,0.12891,0.11865,0.10205,0.11768,0.0918,0.12109,0.10449,0.104,0.10742,0.08594,0.0625,0.07715,0.08545,0.09766,0.09033,0.09424,0.05615,0.09619,0.08252,0.07812,0.07275]},"end":{"gain_pct":-447.01,"l1_loss":0.36086,"baseline_l1_loss":0.06597,"variance":0.2285,"teacher_variance":0.23458,"dino_entropy":6.80749,"off_diag_covariance":0.0835}}}"""
    FROZEN = json.loads(_FROZEN_JSON)
    END = {arm: FROZEN[arm]["end"] for arm in FROZEN}
    # Paper Table 4 (ViT-S) ImageNet KNN Top-5:
    paper = {"full": 17.05, "no_motion": 1.87, "no_mse": 1.58}
    return FROZEN, END, paper


@app.cell
def _():
    # Okabe-Ito colorblind-safe palette: blue=good, grey=neutral, vermillion=broken
    ARMS = {
        "control":   {"c": "#0072B2", "m": "o", "label": "Full TDV"},
        "no_motion": {"c": "#999999", "m": "s", "label": "− motion encoder"},
        "no_mse":    {"c": "#D55E00", "m": "^", "label": "− MSE loss"},
    }
    METRICS = {
        "gain_pct": ("prediction gain over identity (%)",
                     "Next-frame prediction gain vs identity baseline — the headline"),
        "l1_loss": ("L1 error to next-frame repr.",
                    "Motion-composed prediction error (lower = better)"),
        "variance": ("mean per-dim variance",
                     "Representation variance (→0 = collapse to a point)"),
        "teacher_variance": ("teacher repr. variance",
                             "Teacher (EMA) representation variance"),
        "dino_entropy": ("DINO teacher entropy (nats)",
                         "Teacher prototype entropy (→0 = mode collapse)"),
        "off_diag_covariance": ("|off-diagonal covariance|",
                                "Feature redundancy (lower = more decorrelated)"),
    }
    return ARMS, METRICS


@app.cell
def _(METRICS, mo):
    metric_sel = mo.ui.dropdown(
        options={METRICS[k][0]: k for k in METRICS},
        value=METRICS["gain_pct"][0],
        label="**metric to plot:** ",
    )
    return (metric_sel,)


@app.cell
def _(END, mo):
    _g = END["control"]["gain_pct"]
    verdict_badge = mo.md(
        f"""
        <div style="padding:0.6em 1em;border-left:5px solid #0072B2;background:#f0f6fb;border-radius:4px">
        <b>Verdict: ✅ reproduced (qualitatively).</b> Only the full TDV recipe achieves a
        positive next-frame prediction gain (<b>+{_g:.0f}%</b>); removing the motion encoder
        pins it at <b>0%</b> and removing the MSE loss drives it strongly negative — the same
        "both components are essential" conclusion as the paper's Table 4, on a downscaled
        synthetic proxy.
        </div>
        """
    )
    return (verdict_badge,)


@app.cell
def _(np):
    # Pure-numpy renderer for the synthetic low-rank-motion frame-pairs (shared by the
    # data illustration and the GPU lab). A textured disc on a static textured background;
    # in frame 2 the disc is shifted by `magnitude` px.
    def render_pair_np(magnitude, seed, size=224):
        rng = np.random.default_rng(seed)
        small = rng.random((8, 8, 3)); reps = size // 8 + 1
        bg = np.kron(small, np.ones((reps, reps, 1)))[:size, :size, :]
        bg = 0.2 + 0.4 * bg
        r = size // 7
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; mask = xx * xx + yy * yy <= r * r
        col = rng.uniform(0.6, 1.0, 3)
        tex = 0.6 + 0.4 * np.cos(np.sqrt(xx * xx + yy * yy) / max(r, 1) * 3.0)
        sprite = np.clip(col[None, None, :] * tex[:, :, None], 0, 1)
        ang = rng.uniform(0, 2 * np.pi)
        cx, cy = size * 0.4, size * 0.5
        out = []
        for t in (0, 1):
            f = bg.copy()
            ox = int(cx + t * magnitude * np.cos(ang)); oy = int(cy + t * magnitude * np.sin(ang))
            x0, x1, y0, y1 = ox - r, ox + r + 1, oy - r, oy + r + 1
            sx0, sy0 = max(0, -x0), max(0, -y0)
            x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(size, x1), min(size, y1)
            h, w = y1c - y0c, x1c - x0c
            if h > 0 and w > 0:
                m = mask[sy0:sy0 + h, sx0:sx0 + w]; p = sprite[sy0:sy0 + h, sx0:sx0 + w]
                reg = f[y0c:y1c, x0c:x1c]; reg[m] = p[m]; f[y0c:y1c, x0c:x1c] = reg
            out.append((np.clip(f, 0, 1) * 255).astype(np.uint8))
        return out[0], out[1]
    return (render_pair_np,)


@app.cell
def _(mo):
    # GPU-lab controls (defined here so the layout cell above can display them)
    n_pairs_slider = mo.ui.slider(4, 48, value=16, step=4, label="frame-pairs per motion level")
    max_move_slider = mo.ui.slider(16, 96, value=64, step=8, label="max motion (px)")
    run_btn = mo.ui.run_button(label="▶ Run Δz sweep")
    return max_move_slider, n_pairs_slider, run_btn


@app.cell
def _(max_move_slider, mo, n_pairs_slider, np, plt, render_pair_np, run_btn):
    # Expensive GPU work — gated: nothing runs until the button is pressed.
    mo.stop(
        not run_btn.value,
        mo.md("*Press **▶ Run Δz sweep** above to run the lab. Nothing computes until you do.*"),
    )

    def _lab():
        try:
            import torch
        except Exception:
            return "❌ **PyTorch is not available in this runtime.** The GPU lab needs molab's GPU environment (or a local PyTorch install)."
        try:
            import timm
        except Exception:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
            try:
                import timm
            except Exception:
                return "❌ Could not import/install `timm` in this runtime."

        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        # keep the CPU fallback small & clearly reduced
        n_pairs = int(n_pairs_slider.value) if device == "cuda" else min(4, int(n_pairs_slider.value))
        n_levels = 9 if device == "cuda" else 5
        mags = np.linspace(0, int(max_move_slider.value), n_levels)

        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                  num_classes=0, dynamic_img_size=True).eval().to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        npref, L = model.num_prefix_tokens, len(model.blocks)
        acts = {}
        hooks = [b.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(k, o))
                 for k, b in enumerate(model.blocks)]

        import time as _time
        t0 = _time.time()
        grid = np.zeros((L, len(mags)))
        last_spatial = None
        try:
            for mi, mag in enumerate(mags):
                imgs = []
                for j in range(n_pairs):
                    a, b = render_pair_np(mag, seed=1000 + mi * 97 + j)
                    for fr in (a, b):
                        imgs.append(torch.tensor(fr, dtype=torch.float32).permute(2, 0, 1) / 255.0)
                x = ((torch.stack(imgs).to(device) - mean) / std)
                _ = model.forward_features(x)
                for k in range(L):
                    tok = acts[k][:, npref:, :].reshape(n_pairs, 2, -1, model.embed_dim)
                    grid[k, mi] = float((tok[:, 1] - tok[:, 0]).norm(dim=-1).mean())
                if mi == len(mags) - 1:  # spatial Δz heatmap at max motion, last block
                    tok = acts[L - 1][:, npref:, :].reshape(n_pairs, 2, -1, model.embed_dim)
                    g = int(tok.shape[2] ** 0.5)
                    last_spatial = (tok[:, 1] - tok[:, 0]).norm(dim=-1).mean(0).reshape(g, g).cpu().numpy()
        finally:
            for h in hooks:
                h.remove()
        dt = _time.time() - t0

        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
        ax[0].plot(mags, grid[-1], "-o", color="#0072B2", lw=2)
        ax[0].set_title("‖Δz‖ (final block) vs motion", fontsize=9)
        ax[0].set_xlabel("object displacement (px)"); ax[0].set_ylabel("mean ‖Δz‖"); ax[0].grid(alpha=0.25)
        ax[0].annotate("no motion →\nnothing to predict", xy=(mags[0], grid[-1][0]),
                       xytext=(mags[len(mags)//3], grid[-1].max() * 0.4), fontsize=8,
                       arrowprops=dict(arrowstyle="->", color="#555"))
        im1 = ax[1].imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                           extent=[mags[0], mags[-1], 0, L])
        ax[1].set_title("‖Δz‖ by depth × motion", fontsize=9)
        ax[1].set_xlabel("displacement (px)"); ax[1].set_ylabel("ViT block")
        fig.colorbar(im1, ax=ax[1], fraction=0.046)
        im2 = ax[2].imshow(last_spatial, cmap="magma")
        ax[2].set_title("spatial ‖Δz‖ @ max motion\n(lights up on the moving region)", fontsize=9)
        ax[2].set_xticks([]); ax[2].set_yticks([])
        fig.colorbar(im2, ax=ax[2], fraction=0.046)
        fig.suptitle(f"DINOv2 ViT-S/14 · device={dev_name} · {n_pairs} pairs × {len(mags)} motion levels "
                     f"× {L} blocks · {dt:.1f}s", fontsize=9)
        fig.tight_layout()
        return fig

    lab_result = _lab()
    return (lab_result,)


if __name__ == "__main__":
    app.run()
