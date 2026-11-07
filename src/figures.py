"""Development-stage figures and cross-checks (CV curves, shrinkage paths,
spline smoothness, GCV/LOOCV shortcut). Development rows only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler


def _lags(p):
    return "lag 1 only" if p == 1 else f"lags 1..{p}"


def _spec(summary, cid):
    r = summary.set_index("candidate_id").loc[cid]
    return r["feature_set"], r["p"]


def cv_curves(summary, selection, fig_path):
    sel = selection.set_index("family")
    fams = [("ridge", "alpha"), ("lasso", "alpha"), ("spline", "alpha"), ("knn", "k"), ("pcr", "n_components")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (fam, prm) in zip(axes.flat, fams):
        fs, p = _spec(summary, sel.loc[fam, "selected_id"])
        s = summary[(summary["family"] == fam) & (summary["feature_set"] == fs) & (summary["p"] == p)].copy()
        s["x"] = s["params"].map(lambda d: d[prm])
        groups = [(None, s)] if fam != "spline" else list(s.groupby(s["params"].map(lambda d: d["n_knots"])))
        for key, gs in groups:
            gs = gs.sort_values("x")
            ax.errorbar(gs["x"], gs["cv_mae"], yerr=gs["cv_se"], marker="o", ms=3, capsize=2,
                        label=None if key is None else f"{key} knots")
        ax.axhline(sel.loc[fam, "one_se_threshold"], ls=":", c="grey", label="best + 1 SE")
        chosen = s.set_index("candidate_id").loc[sel.loc[fam, "selected_id"]]
        ax.scatter([chosen["x"]], [chosen["cv_mae"]], s=120, facecolors="none", edgecolors="r",
                   zorder=5, label="selected (one-SE)")
        if prm == "alpha":
            ax.set_xscale("log")
        ax.set(title=f"{fam}: {fs}, {_lags(p)}", xlabel=prm, ylabel="CV station-macro MAE (ppb)")
        ax.legend(fontsize=7)
    # last panel: best CV MAE per family x feature set
    ax = axes.flat[-1]
    best = summary.groupby(["family", "feature_set"])["cv_mae"].min().unstack()
    best = best.drop(index=[f for f in ("persistence",) if f in best.index], errors="ignore")
    for i, (fam, row) in enumerate(best.iterrows()):
        ax.plot(row.index, row.values, marker="o", label=fam)
    pers = summary.loc[summary["family"] == "persistence", "cv_mae"].iloc[0]
    ax.axhline(pers, c="k", ls="--", label="persistence")
    ax.set(title="Best CV MAE by feature set", ylabel="CV station-macro MAE (ppb)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150); plt.close(fig)


def shrinkage_paths(dev, cand_by_id, selection, fig_path):
    """Standardized coefficient paths vs penalty on all development rows, with ridge df(alpha)."""
    sel = selection.set_index("family")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    out = {}
    for ax, fam, grid in [(axes[0], "ridge", np.logspace(-2, 6, 60)),
                          (axes[1], "lasso", np.logspace(-4, 1, 60))]:
        c = cand_by_id[sel.loc[fam, "selected_id"]]
        X = StandardScaler().fit_transform(dev[c["columns"]])
        y = dev["y"].to_numpy()
        model = (lambda a: Ridge(alpha=a)) if fam == "ridge" else (lambda a: Lasso(alpha=a, max_iter=50_000))
        coefs = np.array([model(a).fit(X, y).coef_ for a in grid])
        for j, col in enumerate(c["columns"]):
            ax.plot(grid, coefs[:, j], label=col)
        ax.axvline(c["params"]["alpha"], c="r", ls="--", lw=1, label="selected alpha")
        ax.set(xscale="log", xlabel="alpha", ylabel="standardized coefficient",
               title=f"{fam} coefficient paths ({c['fs']}, {_lags(c['p'])})")
        ax.legend(fontsize=6, ncol=2)
        if fam == "ridge":
            d = np.linalg.svd(X - X.mean(0), compute_uv=False)
            df = np.array([(d ** 2 / (d ** 2 + a)).sum() for a in grid])
            axes[2].plot(grid, df)
            axes[2].axvline(c["params"]["alpha"], c="r", ls="--", lw=1)
            axes[2].set(xscale="log", xlabel="alpha", ylabel="effective degrees of freedom",
                        title="Ridge df(alpha) = sum d_j^2 / (d_j^2 + alpha)")
            out["ridge_df_at_selected"] = float((d ** 2 / (d ** 2 + c["params"]["alpha"])).sum())
        else:
            final = Lasso(alpha=c["params"]["alpha"], max_iter=50_000).fit(X, y).coef_
            out["lasso_zeroed_features"] = [col for col, b in zip(c["columns"], final) if b == 0]
    fig.tight_layout(); fig.savefig(fig_path, dpi=150); plt.close(fig)
    return out


def ridge_loocv_crosscheck(dev, cand_by_id, selection, cfg):
    """sklearn RidgeCV uses the closed-form LOOCV shortcut (Class17). Rows are not
    exchangeable here, so this is a sanity check only, not used for selection."""
    c = cand_by_id[selection.set_index("family").loc["ridge", "selected_id"]]
    X = StandardScaler().fit_transform(dev[c["columns"]])
    rcv = RidgeCV(alphas=cfg["grids"]["ridge_alpha"]).fit(X, dev["y"])
    return {"ridge_cv_selected_alpha": c["params"]["alpha"], "ridge_loocv_alpha": float(rcv.alpha_)}


def spline_smoothness(dev, cand_by_id, selection, fig_path):
    """Tomorrow vs today: penalized cubic regression spline at small / selected /
    large alpha, and a classical smoothing spline with lambda chosen by GCV."""
    c = cand_by_id[selection.set_index("family").loc["spline", "selected_id"]]
    x, y = dev["lag1"].to_numpy(), dev["y"].to_numpy()
    grid = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 200)
    st = SplineTransformer(n_knots=c["params"]["n_knots"], degree=3, knots="quantile",
                           extrapolation="linear").fit(x[:, None])
    B, Bg = st.transform(x[:, None]), st.transform(grid[:, None])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    lim = (0, np.percentile(np.r_[x, y], 99.9))
    for ax in axes:
        ax.hexbin(x, y, gridsize=60, bins="log", cmap="Greys", mincnt=1, extent=lim + lim)
        ax.set(xlim=lim, ylim=lim)
        ax.plot(grid, grid, c="k", ls=":", lw=1, label="persistence (y = x)")
    a_sel = c["params"]["alpha"]
    for a, lab in [(a_sel / 1000, "small alpha"), (a_sel, "selected alpha"), (a_sel * 1e4, "large alpha")]:
        axes[0].plot(grid, Ridge(alpha=a).fit(B, y).predict(Bg), lw=2, label=f"{lab} ({a:g})")
    axes[0].set(title=f"Cubic regression spline on lag1, {c['params']['n_knots']} knots (univariate view)",
                xlabel="today's MDA8 (ppb)", ylabel="tomorrow's MDA8 (ppb)")
    axes[0].legend(fontsize=8)
    # smoothing spline needs strictly increasing x: average y at each distinct x, weight by count
    agg = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["mean", "size"])
    ss = make_smoothing_spline(agg.index.to_numpy(float), agg["mean"].to_numpy(), w=agg["size"].to_numpy(float))
    axes[1].plot(grid, ss(grid), c="C3", lw=2, label="smoothing spline, lambda by GCV")
    axes[1].set(title="Smoothing spline (scipy make_smoothing_spline, GCV)",
                xlabel="today's MDA8 (ppb)", ylabel="tomorrow's MDA8 (ppb)")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150); plt.close(fig)
