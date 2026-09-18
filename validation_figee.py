"""Validation d'un jeu de paramètres FIGÉ (celui qui passe en production).

    from dav_runoff.validation_figee import valider_parametres_figes

    res = valider_parametres_figes(etude, bor3m, output_dir="dav_out_full_rr_v6", modele="M1",
                                   date_coupure="2023-10", **COLONNES)
    res["fige_par_bucket"]     # backtest statique : theta_hat appliqué à toutes les cohortes, RMSE par bucket × horizon
    res["fige_par_origine"]    # RMSE par origine
    res["holdout"]             # theta estimé sur les données <= date_coupure, figé, RMSE sur les cohortes postérieures
    res["theta_holdout"]       # le jeu figé du hold-out (à comparer à theta_hat)
    res["convergence"]         # écart relatif entre theta_hat et les dernières estimations glissantes (stabilite.csv)
    res["seuils_suivi"]        # seuils de suivi mensuel proposés (RMSE et dérive des paramètres)

Deux tests complémentaires du backtest glissant, qui valide la PROCÉDURE : ici on valide le JEU DE NOMBRES.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import features
from .backtest import regime_of_origin
from .config import MODELS, PARAM_NAMES, Config
from .estimation import RetentionModel, fit
from .ingestion import Source, load
from .preprocessing import build_panel
from .target import age_bucket_labels, retention_surface


def _rmse_table(cells: pd.DataFrame, pred: np.ndarray, cfg: Config) -> pd.DataFrame:
    d = cells[["t0", "h", "regime_test"]].copy()
    d["e2"] = (cells.rho_A.to_numpy() - pred) ** 2
    d["bucket"] = age_bucket_labels(d.h, cfg.age_buckets)
    tab = d.groupby(["regime_test", "bucket"]).e2.mean().pow(0.5).unstack("bucket")
    tab["global"] = d.groupby("regime_test").e2.mean().pow(0.5)
    tab["n_origines"] = d.groupby("regime_test").t0.nunique()
    tot = d.e2.mean() ** 0.5
    tab.loc["toutes"] = {**d.assign(b=d.bucket).groupby("b").e2.mean().pow(0.5).to_dict(), "global": tot, "n_origines": d.t0.nunique()}
    return tab


def valider_parametres_figes(etude: Source, bor3m: Source, output_dir: str, modele: str = "M1",
                             date_coupure: Optional[str] = "2023-10", **load_kwargs) -> Dict[str, object]:
    journal = json.load(open(os.path.join(output_dir, "journal.json"), encoding="utf-8"))
    mode = journal["constantes"].get("memoire_choc", "calendaire")
    cfg = Config(cohort_reset=("cohorte" in mode), level_relative=("relatif" in mode))
    r_cap, dr_ref = journal["constantes"]["r_cap"], journal["constantes"]["dr_ref"]
    params = pd.read_csv(os.path.join(output_dir, "parametres.csv"), index_col=0)
    theta_hat = {k: float(params.loc[modele, k]) for k in PARAM_NAMES}

    panel = build_panel(load(etude, bor3m, **load_kwargs), cfg)
    rm = RetentionModel(panel.r_full, r_cap, dr_ref, cohort_reset=cfg.cohort_reset, level_relative=cfg.level_relative)
    y = features.young_share_series(panel.B, cfg.seuil_jeune)
    cells = retention_surface(panel, cfg.h_est).estimation_cells(cfg.h_est)
    cells["regime_test"] = [regime_of_origin(rm.r, panel.offset, int(t0), cfg.regime_min_train_months) for t0 in cells.t0]
    t0 = cells.t0.to_numpy().astype(int)
    h = cells.h.to_numpy().astype(int)
    out: Dict[str, object] = {"theta_hat": theta_hat}

    # ---- 1. backtest statique : theta_hat figé appliqué à toutes les cohortes (chemin de taux réalisé)
    pred = rm.predict(theta_hat, t0 + panel.offset, h, y[t0])
    out["fige_par_bucket"] = _rmse_table(cells, pred, cfg)
    po = cells.assign(e2=(cells.rho_A.to_numpy() - pred) ** 2).groupby(["origine", "regime_test"]).e2.mean().pow(0.5).rename("rmse").reset_index()
    out["fige_par_origine"] = po

    # ---- 2. hold-out temporel : estimation <= date_coupure, figée, prédiction des cohortes postérieures
    if date_coupure is not None:
        tc = int(panel.months.get_loc(pd.Period(date_coupure, "M")))
        train = cells[cells.cal <= tc]
        test = cells[cells.t0 > tc]
        f = fit(modele, train, y, rm, cfg, panel.offset, n_starts=cfg.backtest_n_starts, theta0=dict(cfg.theta_init))
        assert (train.t0 + train.h <= tc).all()
        if not test.empty:
            pt = rm.predict(f.theta, test.t0.to_numpy().astype(int) + panel.offset, test.h.to_numpy().astype(int), y[test.t0.to_numpy().astype(int)])
            d = test.assign(e2=(test.rho_A.to_numpy() - pt) ** 2)
            hold = d.groupby("h").e2.mean().pow(0.5).rename("rmse").to_frame()
            hold.loc["global"] = d.e2.mean() ** 0.5
            out["holdout"] = hold
            out["holdout_n_origines"] = int(test.t0.nunique())
        out["theta_holdout"] = f.theta
        out["ecart_theta_holdout_vs_hat"] = {p: (f.theta[p] - theta_hat[p]) / max(abs(theta_hat[p]), 1e-8) for p in MODELS[modele]}

    # ---- 3. convergence des estimations glissantes vers theta_hat
    st_path = os.path.join(output_dir, "stabilite.csv")
    if os.path.exists(st_path):
        st = pd.read_csv(st_path)
        last = st.tail(3)
        out["convergence"] = pd.DataFrame({p: [(last[p].mean() - theta_hat[p]) / max(abs(theta_hat[p]), 1e-8)] for p in MODELS[modele] if p in st.columns},
                                          index=["ecart_relatif_3_dernieres_fenetres"])

    # ---- 4. seuils de suivi proposés
    ref = out["fige_par_bucket"]
    base_rmse = float(ref.loc["positif", "global"]) if "positif" in ref.index else float(ref.loc["toutes", "global"])
    out["seuils_suivi"] = {"rmse_12m_alerte": round(base_rmse + 0.05, 3),
                           "derive_parametre_alerte_%": 20,
                           "regle": "ré-estimer si le RMSE mensuel à 12 mois dépasse le seuil deux mois de suite, "
                                    "ou si une ré-estimation déplace lam_s ou a1 de plus de 20 %"}
    return out


# ============================================================================ analyse de drift à paramètres figés

def dates_regimes(panel) -> Dict[str, Optional[str]]:
    """Repères de la courbe de taux sur la période du panel : début de la montée (premier mois où le taux a monté d'au moins 25 pb
    sur 3 mois après une période plate/négative), passage en positif, sommet, début de la descente
    (premier mois où le taux a baissé d'au moins 25 pb depuis le sommet)."""
    months = panel.months
    r = np.asarray(panel.r_full[panel.offset:panel.offset + panel.T], dtype=float)
    out: Dict[str, Optional[str]] = {"debut_montee": None, "passage_positif": None, "sommet": None, "debut_descente": None}
    for t in range(3, len(r)):
        if r[t] - r[t - 3] >= 0.25:
            out["debut_montee"] = str(months[t - 3]); break
    pos = np.where(r > 0)[0]
    if len(pos):
        out["passage_positif"] = str(months[int(pos[0])])
    tmax = int(np.argmax(r)); out["sommet"] = str(months[tmax])
    for t in range(tmax + 1, len(r)):
        if r[tmax] - r[t] >= 0.25:
            out["debut_descente"] = str(months[t]); break
    return out


def analyser_drift(etude: Source, bor3m: Source, output_dir: str, date_coupure: str, modele: str = "M1",
                   seuil_mois: float = 2.0, n_consecutifs: int = 2, h_fixe: int = 12, h_max: int = 60,
                   label: Optional[str] = None, **load_kwargs) -> Dict[str, object]:
    """Le modèle est estimé sur les seules données antérieures à date_coupure, ses paramètres sont FIGÉS, puis
    il prédit chaque cohorte postérieure. Pour chacune on compare la WAL observée (sur tous les mois observés
    disponibles, jusqu'à h_max) et la WAL du modèle figé sur le même horizon ; idem à horizon fixe h_fixe pour
    une lecture comparable d'une cohorte à l'autre. Le drift est déclaré à la première cohorte à partir de
    laquelle l'écart dépasse seuil_mois pendant n_consecutifs cohortes consécutives (à horizon fixe).
    Renvoie le tableau par cohorte, la synthèse (mois avant drift, RMSE par régime) et trace le graphique."""
    import matplotlib.pyplot as plt

    journal = json.load(open(os.path.join(output_dir, "journal.json"), encoding="utf-8"))
    mode = journal["constantes"].get("memoire_choc", "calendaire")
    cfg = Config(cohort_reset=("cohorte" in mode), level_relative=("relatif" in mode))
    r_cap, dr_ref = journal["constantes"]["r_cap"], journal["constantes"]["dr_ref"]
    panel = build_panel(load(etude, bor3m, **load_kwargs), cfg)
    months = [str(m) for m in panel.months]
    tc = months.index(str(pd.Period(date_coupure, "M")))
    rm = RetentionModel(panel.r_full, r_cap, dr_ref, cohort_reset=cfg.cohort_reset, level_relative=cfg.level_relative)
    rm_ext = RetentionModel(np.concatenate([panel.r_full, np.full(h_max + 1, float(panel.r_full[-1]))]), r_cap, dr_ref,
                            cohort_reset=cfg.cohort_reset, level_relative=cfg.level_relative)
    y = features.young_share_series(panel.B, cfg.seuil_jeune)
    cells = retention_surface(panel, cfg.h_est).estimation_cells(cfg.h_est)
    train = cells[cells.cal <= tc]
    assert (train.t0 + train.h <= tc).all()
    f = fit(modele, train, y, rm, cfg, panel.offset, n_starts=cfg.n_starts, theta0=dict(cfg.theta_init))
    theta = f.theta
    surf = retention_surface(panel, h_max).cells
    regimes = [regime_of_origin(rm.r, panel.offset, int(t), cfg.regime_min_train_months) for t in range(panel.T)]

    rows = []
    for t0 in range(panel.T - 1):
        d = surf[surf.t0 == t0].set_index("h").sort_index().rho_A
        H = min(h_max, panel.T - 1 - t0)
        pred = rm_ext.predict_curve(theta, panel.rate_index(t0), float(y[t0]), h_max).set_index("h").rho
        w_obs = 1.0 + float(d.reindex(range(1, H + 1)).sum()) if H >= 1 else np.nan
        w_pred = 1.0 + float(pred.reindex(range(1, H + 1)).sum())
        obs12 = d.reindex(range(1, h_fixe)).to_numpy(dtype=float)
        rows.append({"origine": months[t0], "mois_depuis_coupure": t0 - tc, "regime": regimes[t0], "H_dispo": H,
                     "bor3m": float(panel.r_full[panel.rate_index(t0)]), "stock_M": float(panel.B[:, t0].sum()) / 1e6, "y": float(y[t0]),
                     "WAL_obs_dispo": w_obs, "WAL_fige_dispo": w_pred,
                     "WAL_obs_fixe": (1.0 + float(obs12.sum())) if not np.isnan(obs12).any() else np.nan,
                     "WAL_fige_fixe": 1.0 + float(pred.reindex(range(1, h_fixe)).sum())})
    w = pd.DataFrame(rows)
    w["ecart_dispo"] = w.WAL_fige_dispo - w.WAL_obs_dispo
    w["ecart_fixe"] = w.WAL_fige_fixe - w.WAL_obs_fixe

    # --- détection du drift (après la coupure, horizon fixe)
    apres = w[(w.mois_depuis_coupure > 0) & w.ecart_fixe.notna()].reset_index(drop=True)
    drift_origine, drift_mois = None, None
    dep = (apres.ecart_fixe.abs() > seuil_mois).to_numpy()
    for i in range(len(dep) - n_consecutifs + 1):
        if dep[i:i + n_consecutifs].all():
            drift_origine = apres.origine.iloc[i]; drift_mois = int(apres.mois_depuis_coupure.iloc[i]); break
    def agg(g):
        return pd.Series({"n": len(g), "WAL_obs_moy": g.WAL_obs_fixe.mean(), "WAL_fige_moy": g.WAL_fige_fixe.mean(),
                          "biais_mois": g.ecart_fixe.mean(), "rmse_mois": float(np.sqrt(np.mean(g.ecart_fixe ** 2)))})
    synth = {"date_coupure": str(months[tc]), "modele": modele, "theta_fige": theta, "n_cohortes_train": int(train.t0.nunique()),
             "premiere_cohorte_predite": months[tc + 1] if tc + 1 < panel.T else None,
             "seuil_mois": seuil_mois, "n_consecutifs": n_consecutifs, "horizon_fixe": h_fixe,
             "drift_detecte_origine": drift_origine, "mois_avant_drift": drift_mois,
             "par_regime_apres_coupure": apres.groupby("regime").apply(agg, include_groups=False).round(2),
             "par_annee_apres_coupure": apres.assign(annee=apres.origine.str[:4]).groupby("annee").apply(agg, include_groups=False).round(2)}
    tag = label or f"drift_{months[tc]}_{modele}"
    w.to_csv(os.path.join(output_dir, f"{tag}.csv"), index=False)

    # --- graphique
    coul = {"negatif": "#c9cfd8", "transition": "#f2d7a7", "positif": "#cfe3d4"}
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.4), sharex=True)
    x = np.arange(len(w))
    for ax, (obs, pred, titre) in zip(axes, [("WAL_obs_dispo", "WAL_fige_dispo", f"WAL sur tous les mois observés disponibles (≤ {h_max}) — même horizon pour observé et modèle"),
                                             ("WAL_obs_fixe", "WAL_fige_fixe", f"WAL tronquée à {h_fixe} mois — comparable d'une cohorte à l'autre")]):
        for i, reg in enumerate(w.regime):
            ax.axvspan(i - 0.5, i + 0.5, color=coul.get(reg, "white"), alpha=0.5, lw=0)
        ax.plot(x, w[obs], color="#1f3b5c", lw=2.2, marker="o", ms=3, label="WAL observée")
        ax.plot(x, w[pred], color="#d1495b", lw=2, ls="--", marker="o", ms=3, label=f"WAL modèle figé à {months[tc]}")
        ax.axvline(tc + 0.5, color="black", lw=1.2, ls="-.")
        ax.annotate("paramètres figés ici →", (tc + 0.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1), fontsize=8, ha="right")
        if drift_origine is not None:
            i_d = months.index(drift_origine)
            ax.axvline(i_d, color="#d1495b", lw=1.2, ls=":")
            ax.annotate(f"drift détecté : {drift_origine} (+{drift_mois} mois)", (i_d, 1.0), fontsize=8, ha="left", color="#d1495b")
        ax.set_ylabel("WAL (mois)"); ax.grid(alpha=0.3); ax.set_title(titre, fontsize=10); ax.legend(fontsize=8, loc="upper right")
    ax2 = axes[0].twinx(); ax2.plot(x, w.bor3m, color="#8a94a6", lw=1, label="BOR3M"); ax2.set_ylabel("BOR3M (%)", color="#8a94a6")
    step = max(1, len(x) // 14)
    axes[1].set_xticks(x[::step]); axes[1].set_xticklabels(w.origine.iloc[::step], rotation=45, fontsize=8)
    fig.suptitle(f"Drift à paramètres constants — modèle {modele} estimé sur les données ≤ {months[tc]} ({train.t0.nunique()} cohortes) puis figé", fontsize=11)
    fig.tight_layout()
    png = os.path.join(output_dir, f"{tag}.png"); fig.savefig(png, dpi=130); print("→", png)
    print(f"coupure {months[tc]} — drift détecté : {drift_origine} (+{drift_mois} mois)" if drift_origine else f"coupure {months[tc]} — aucun drift au seuil de {seuil_mois} mois")
    print(synth["par_regime_apres_coupure"].to_string())
    return {"par_cohorte": w, "synthese": synth}
