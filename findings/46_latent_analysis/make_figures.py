import glob, json, os
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
# Figures for findings/46 (latent analysis, pass 1) from raw/latent_*.json and raw/cka_d*.json.  Run inside a kernel where
# figure-style's apply_figure_style / panel_letter are loaded: exec(open(<this file>).read()).
HERE = os.path.join(os.environ["PCP_ROOT"], "findings/46_latent_analysis"); RAW = HERE + "/raw"
L = {}
for f in glob.glob(RAW + "/latent_*.json"):
    r = json.load(open(f)); L[r["run"]] = r
CK = {d: json.load(open(f"{RAW}/cka_d{d}.json")) for d in (6, 10)}
FLOOR = json.load(open(RAW + "/ctx_floor.json"))
MODELS = [("bp", "Backprop"), ("bp_lrmatch", "Backprop, lr 1e-3"), ("epc_alm_blk4", "ePC + mult., per-block ρ"),
          ("epc_alm_k4", "ePC + mult., median ρ"), ("epc_k4", "ePC, no mult. (lr 1e-3)"), ("init", "Untrained")]
COL = {"bp": "#222222", "bp_lrmatch": "#999999", "epc_alm_blk4": "#0072B2", "epc_alm_k4": "#E69F00", "epc_k4": "#009E73", "init": "#BBBBBB"}
LS = {"bp": "-", "bp_lrmatch": "--", "epc_alm_blk4": "-", "epc_alm_k4": "-", "epc_k4": "-", "init": ":"}
LAB = dict(MODELS)


def runs(depth, arm):
    return sorted([r for r in L.values() if r["depth"] == depth and r["arm"] == arm], key=lambda r: r["seed"])


def val(depth, arm, fn):                       # mean, min, max over seeds (BP: 5 seeds; others: 1)
    v = np.array([fn(r) for r in runs(depth, arm)], dtype=float)
    return v.mean(0), v.min(0), v.max(0)


def save(fig, name):
    fig.savefig(f"{HERE}/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{HERE}/{name}.pdf", bbox_inches="tight")


apply_figure_style(sizes=(8, 7, 6))
TRAINED = [m for m in MODELS if m[0] != "init"]

# ---------------------------------------------------------------- Figure 1: noise
fig = plt.figure(figsize=(7.6, 2.8)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.5)
ax = fig.add_subplot(gs[0]); y = 0; yt, yl = [], []
for depth in (10, 6):
    ax.text(0.0, y, f"depth {depth}", fontsize=7, color="#555555", fontweight="bold", va="center", transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    y -= 1
    for arm, lab in TRAINED:
        m_c, lo_c, hi_c = val(depth, arm, lambda r: r["conds"]["noise"]["noise"]["nmse_cal_cleanalpha"])
        m_n, *_ = val(depth, arm, lambda r: r["conds"]["noise"]["nmse_cal"])
        ax.plot([m_n, m_c], [y, y], color=COL[arm], lw=0.8, alpha=0.6, zorder=1)
        ax.scatter([m_c], [y], s=22, color=COL[arm], zorder=3)
        ax.scatter([m_n], [y], s=22, facecolor="white", edgecolor=COL[arm], lw=1.0, zorder=3)
        if arm == "bp": ax.plot([lo_c, hi_c], [y, y], color=COL[arm], lw=2.2, solid_capstyle="butt", zorder=2)
        yt.append(y); yl.append(lab); y -= 1
    y -= 0.8
ax.axvline(1.0, color="#BBBBBB", lw=0.8, ls=":", zorder=0)
ax.text(1.003, -0.2, "untrained\npredictor", fontsize=6, color="#777777", va="center")
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(y + 0.4, 0.6)
ax.set_xlabel("test error under noise (nmse)")
ax.scatter([], [], s=22, color="#444444", label="scale fitted on clean images"); ax.scatter([], [], s=22, facecolor="white", edgecolor="#444444", label="scale refit on noisy images")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=6, ncol=1, handletextpad=0.3)
ax.set_title("With the clean-image scale, PC is more robust", loc="left", pad=28)
panel_letter(ax, "a", dx=-0.62)
ax = fig.add_subplot(gs[1])
for depth, mk in ((10, "o"), (6, "s")):
    for arm, lab in TRAINED:
        for r in runs(depth, arm):
            n = r["conds"]["noise"]["noise"]
            ax.scatter(n["rel_change"], n["nmse_cal_cleanalpha"], marker=mk, s=16, color=COL[arm], alpha=0.85, lw=0)
ax.set_xlabel("prediction change under noise\n‖ΔP‖² / ‖P‖²"); ax.set_ylabel("test error under noise (nmse)")
ax.scatter([], [], marker="o", s=16, color="#666666", label="depth 10"); ax.scatter([], [], marker="s", s=16, color="#666666", label="depth 6")
ax.legend(loc="upper left", fontsize=6); ax.margins(0.08)
ax.set_title("Error follows how far\npredictions move", loc="left"); panel_letter(ax, "b")
ax = fig.add_subplot(gs[2])
for depth, mk in ((10, "o"), (6, "s")):
    for arm, lab in TRAINED:
        for r in runs(depth, arm):
            ax.scatter(r["sensitivity"]["jac_top_x_alpha_med"], r["conds"]["noise"]["noise"]["noise_gain_x_alpha_med"], marker=mk, s=16, color=COL[arm], alpha=0.85, lw=0)
ax.set_xscale("log"); ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([2, 3, 4, 6])); ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel("worst-case sensitivity\n(top singular value of ∂P/∂c)"); ax.set_ylabel("sensitivity along real noise\n‖ΔP‖ / ‖Δc‖")
ax.margins(0.08); ax.set_title("PC: larger worst case,\nsmaller along real noise", loc="left"); panel_letter(ax, "c")
for arm, lab in TRAINED[::-1]:
    fig.axes[1].plot([], [], color=COL[arm], lw=3, label=lab)
h, l = fig.axes[1].get_legend_handles_labels()
fig.legend(h[2:][::-1], l[2:][::-1], loc="lower center", bbox_to_anchor=(0.62, -0.17), ncol=5, fontsize=6, handlelength=1.2)
save(fig, "fig1_noise"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: missing context
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.45})
ax = axs[0]; xs = np.arange(3); w = 0.15
for i, (arm, lab) in enumerate(TRAINED):
    for depth, alpha in ((10, 1.0),):
        m, lo, hi = val(depth, arm, lambda r: [r["interventions"][k]["delta"] for k in ("near", "random", "far")])
        ax.bar(xs + (i - 2) * w, m, w * 0.92, color=COL[arm], label=lab)
        if arm == "bp": ax.errorbar(xs + (i - 2) * w, m, yerr=[m - lo, hi - m], fmt="none", ecolor="#666666", elinewidth=0.8, capsize=1.5)
ax.set_xticks(xs); ax.set_xticklabels(["nearby\n(≤1.5 patches)", "same number,\nrandom", "far\n(>2.1 patches)"])
ax.set_ylabel("increase in test error (nmse)"); ax.axhline(0, color="#999999", lw=0.6)
ax.set_title("Nearby context matters more\nto backprop (depth 10)", loc="left"); panel_letter(ax, "a", dx=-0.25)
ax = axs[1]; xd = np.array([1.0, 1.41, 2.0, 3.0, 4.0]); xl = ["1", "√2", "2", "3", ">3"]
for arm, lab in TRAINED:
    m, lo, hi = val(10, arm, lambda r: r["conds"]["clean"]["dist_nmse"])
    ax.plot(xd, m, ls=LS[arm], color=COL[arm], marker="o", ms=2.5, lw=1.2)
ax.set_xticks(xd); ax.set_xticklabels(xl); ax.set_xlabel("distance of target patch to\nnearest visible context patch")
ax.set_ylabel("test error (nmse)"); ax.margins(0.06)
ax.set_title("Error rises away from\ncontext (depth 10)", loc="left"); panel_letter(ax, "b")
ax = axs[2]
for arm, lab in TRAINED:
    m, lo, hi = val(10, arm, lambda r: [a[1] / a[2] for a in r["conds"]["clean"]["attn_layer_mean"]])
    lay = np.arange(1, len(m) + 1); ax.plot(lay, m, ls=LS[arm], color=COL[arm], marker="o", ms=2.5, lw=1.2)
    if arm == "bp": ax.fill_between(lay, lo, hi, color=COL[arm], alpha=0.15, lw=0)
ax.axhline(1.0, color="#BBBBBB", lw=0.8, ls=":"); ax.text(10, 1.0, "uniform attention", fontsize=6, color="#777777", ha="right", va="bottom")
ax.set_xlabel("attention layer (depth 10)"); ax.set_ylabel("attended distance to context\n÷ distance under uniform attention")
ax.set_title("PC's early layers attend\nless locally", loc="left"); ax.margins(0.05); panel_letter(ax, "c")
axs[0].legend(loc="upper right", bbox_to_anchor=(1.03, 0.8), fontsize=6, handlelength=1.0)
save(fig, "fig2_context"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: what clean error is made of
BN = ["top 50%", "50–80%", "80–95%", "last 5%"]
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7), gridspec_kw={"wspace": 0.5})
ax = axs[0]
for arm, lab in TRAINED:
    m, lo, hi = val(10, arm, lambda r: [b["err_over_target_var"] for b in r["conds"]["clean"]["bands"][:4]])
    ax.plot(range(4), m, ls=LS[arm], color=COL[arm], marker="o", ms=3, lw=1.2, label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(BN); ax.set_xlabel("target directions\n(bands of cumulative variance)")
ax.set_ylabel("error ÷ target variance\nin those directions"); ax.margins(0.06)
ax.set_title("PC loses most in the\nfine directions (depth 10)", loc="left"); panel_letter(ax, "a", dx=-0.3)
ax.legend(loc="upper left", fontsize=6, handlelength=1.2, bbox_to_anchor=(-0.02, 1.02))
ax = axs[1]; ev = np.load(RAW + "/target_basis.npz")["ev"]
def smooth(x, k=9):
    x = np.asarray(x[:191]); return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
b0, blo, bhi = val(10, "bp", lambda r: smooth(r["conds"]["clean"]["per_comp"]["slope"]))
ax.fill_between(np.arange(1, 192), blo - b0, bhi - b0, color=COL["bp"], alpha=0.25, lw=0)
ax.text(1.2, 0.006, "backprop seed range", fontsize=6, color="#555555", va="bottom")
for arm, lab in TRAINED[1:]:
    m, lo, hi = val(10, arm, lambda r: smooth(r["conds"]["clean"]["per_comp"]["slope"]))
    ax.plot(np.arange(1, 192), m - b0, ls=LS[arm], color=COL[arm], lw=1.1)
ax.axhline(0, color=COL["bp"], lw=0.8)
ax.set_xscale("log"); ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("target direction rank\n(1 = most variance)"); ax.set_ylabel("slope minus backprop's slope")
ax.set_title("PC under-predicts the\nlow-variance directions", loc="left"); ax.margins(0.04); panel_letter(ax, "b")
ax = axs[2]
for arm, lab in TRAINED[1:]:
    g = {}
    for depth in (6, 10):
        b, *_ = val(depth, "bp", lambda r: [x["err_over_target_var"] for x in r["conds"]["clean"]["bands"][:4]])
        p, *_ = val(depth, arm, lambda r: [x["err_over_target_var"] for x in r["conds"]["clean"]["bands"][:4]])
        g[depth] = p - b
    for k, mk in ((0, "o"), (3, "^")):
        ax.plot([6, 10], [g[6][k], g[10][k]], color=COL[arm], ls=LS[arm], marker=mk, ms=4, lw=1.1, mfc=COL[arm] if k == 3 else "white")
ax.axhline(0, color="#999999", lw=0.6); ax.set_xticks([6, 10]); ax.set_xlabel("predictor depth"); ax.set_xlim(5, 11)
ax.set_ylabel("gap to backprop\n(error ÷ target variance)")
ax.plot([], [], marker="^", color="#555555", ls="none", label="last 5% of variance"); ax.plot([], [], marker="o", mfc="white", color="#555555", ls="none", label="top 50% of variance")
ax.legend(loc="center right", fontsize=6); ax.margins(0.06)
ax.set_title("Depth narrows the fine-detail\ngap for the multiplier arms", loc="left"); panel_letter(ax, "c")
save(fig, "fig3_error_directions"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: internal solution and class information
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7), gridspec_kw={"width_ratios": [1, 1.4, 1], "wspace": 0.42})
def pair(ck, a, b):
    return ck["pairs"].get(f"{a}|{b}") or ck["pairs"].get(f"{b}|{a}")
for ax, depth, letter in ((axs[0], 6, "a"), (axs[1], 10, "b")):
    ck = CK[depth]; bps = [r["run"] for r in runs(depth, "bp")]
    bb = np.array([pair(ck, a, b) for i, a in enumerate(bps) for b in bps[i + 1:]]); nodes = np.arange(1, bb.shape[1] + 1)
    ax.fill_between(nodes, bb.min(0), bb.max(0), color=COL["bp"], alpha=0.25, lw=0); ax.plot(nodes, bb.mean(0), color=COL["bp"], lw=1.2)
    for arm, lab in MODELS[1:]:
        rr = runs(depth, arm)
        if not rr: continue
        v = np.array([pair(ck, rr[0]["run"], b) for b in bps[1:]]).mean(0)
        ax.plot(nodes, v, color=COL[arm], ls=LS[arm], marker="o", ms=2, lw=1.1)
    ax.set_xlabel(f"layer node (depth {depth})"); ax.set_ylim(0.38, 1.03); ax.margins(x=0.04)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    if depth == 6: ax.set_ylabel("similarity to backprop seeds 1–4\n(linear CKA)")
    ax.set_title("Depth 6: PC's middle and late\nlayers differ from backprop" if depth == 6 else "Depth 10: PC differs in the middle,\nre-converges near the output", loc="left")
    panel_letter(ax, letter, dx=-0.3 if depth == 6 else -0.12)
axs[1].text(13.2, 0.995, "backprop vs backprop", fontsize=6, color=COL["bp"], va="bottom")
for arm, lab in MODELS[1:]:
    axs[1].plot([], [], color=COL[arm], ls=LS[arm], lw=1.2, label=lab)
axs[1].legend(loc="lower left", fontsize=6, handlelength=1.4)
ax = axs[2]
for arm, lab in MODELS:
    m, lo, hi = val(10, arm, lambda r: [x["ridge"] for x in r["node_probes"]][1:])
    ax.plot(np.arange(1, len(m) + 1), m, color=COL[arm], ls=LS[arm], lw=1.1)
ax.axhline(FLOOR["ridge"], color="#D55E00", lw=0.8, ls="--"); ax.text(20, FLOOR["ridge"] + 0.002, "context embedding alone", fontsize=6, color="#D55E00", ha="right", va="bottom")
ax.set_xlabel("layer node, depth 10"); ax.set_ylabel("linear-probe class accuracy"); ax.set_ylim(0.47, 0.645); ax.margins(x=0.04)
ax.set_title("Hidden states hold no more class\ninformation than the context", loc="left"); panel_letter(ax, "c")
save(fig, "fig4_internal_solution"); plt.close(fig)
print("figures written")
