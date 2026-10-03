"""
Modèle de Markowitz appliqué à des stratégies (et non à des titres).

Étapes, dans l'ordre du cours :
  1. rendement espéré, risque (écart-type) et corrélations de chaque stratégie ;
  2. ensemble des portefeuilles possibles et frontière efficiente ;
  3. actif sans risque -> droite d'allocation du capital (CAL) et portefeuille risqué optimal
     (portefeuille "tangent", celui qui maximise le ratio de Sharpe) ;
  4. préférences de l'investisseur (aversion au risque A, courbes d'indifférence)
     -> portefeuille optimal complet : y* = (E(r_T) - r_f) / (A * sigma_T^2) dans le portefeuille
     risqué, le reste au taux sans risque.

Utilité utilisée : U = E(r) - 1/2 * A * sigma^2 (rendements et variances en décimal, annualisés).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PERIODS = {"Hebdomadaire": ("W-FRI", 52), "Quotidienne": ("B", 252), "Mensuelle": ("ME", 12)}


# ---------------------------------------------------------------------------
# 1. DONNÉES
# ---------------------------------------------------------------------------

def aligned_returns(perfs: dict[str, pd.Series], freq: str = "W-FRI") -> pd.DataFrame:
    """
    perfs : {nom: performance cumulée (0.12 = +12 %)} indexée par date.
    Renvoie les rendements périodiques sur la période commune à toutes les stratégies.
    """
    levels = {}
    for name, p in perfs.items():
        s = (1 + p.astype(float)).sort_index()
        s.index = pd.to_datetime(s.index)
        levels[name] = s.resample(freq).last()
    df = pd.DataFrame(levels)
    start = max(s.first_valid_index() for s in levels.values())
    end = min(s.last_valid_index() for s in levels.values())
    df = df.loc[start:end].ffill()
    return df.pct_change().dropna(how="any")


@dataclass
class Inputs:
    names: list[str]
    mu: np.ndarray        # rendements espérés annualisés
    cov: np.ndarray       # matrice de covariance annualisée
    sigma: np.ndarray     # volatilités annualisées
    corr: pd.DataFrame
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp


def estimate(returns: pd.DataFrame, periods_per_year: int) -> Inputs:
    mu = returns.mean().values * periods_per_year
    cov = returns.cov().values * periods_per_year
    sigma = np.sqrt(np.diag(cov))
    return Inputs(list(returns.columns), mu, cov, sigma, returns.corr(),
                  len(returns), returns.index.min(), returns.index.max())


# ---------------------------------------------------------------------------
# 2. PORTEFEUILLES
# ---------------------------------------------------------------------------

def port_stats(w: np.ndarray, inp: Inputs, rf: float = 0.0) -> dict:
    ret = float(w @ inp.mu)
    vol = float(np.sqrt(max(w @ inp.cov @ w, 0.0)))
    return {"ret": ret, "vol": vol, "sharpe": (ret - rf) / vol if vol > 0 else np.nan}


def _bounds(n: int, allow_short: bool):
    return [(-1.0, 2.0) if allow_short else (0.0, 1.0)] * n


def _solve(fun, n, allow_short, extra_cons=()):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_cons]
    best = None
    for x0 in [np.full(n, 1 / n)] + [np.eye(n)[i] for i in range(n)]:
        res = minimize(fun, x0, method="SLSQP", bounds=_bounds(n, allow_short),
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Optimisation impossible avec ces paramètres.")
    w = best.x
    w[np.abs(w) < 1e-6] = 0.0
    return w / w.sum()


def min_variance(inp: Inputs, allow_short: bool = False) -> np.ndarray:
    """Portefeuille de variance minimale (point le plus à gauche de la frontière)."""
    return _solve(lambda w: w @ inp.cov @ w, len(inp.mu), allow_short)


def max_sharpe(inp: Inputs, rf: float, allow_short: bool = False) -> np.ndarray:
    """Portefeuille risqué optimal : point de tangence entre la CAL et la frontière."""
    def neg_sharpe(w):
        vol = np.sqrt(max(w @ inp.cov @ w, 1e-18))
        return -(w @ inp.mu - rf) / vol
    return _solve(neg_sharpe, len(inp.mu), allow_short)


def frontier(inp: Inputs, allow_short: bool = False, points: int = 60) -> pd.DataFrame:
    """Frontière efficiente : variance minimale pour chaque niveau de rendement visé."""
    n = len(inp.mu)
    w_min = min_variance(inp, allow_short)
    r_min = float(w_min @ inp.mu)
    r_max = float(inp.mu.max()) if not allow_short else float(inp.mu.max()) * 1.5 + 1e-9
    rows = []
    for target in np.linspace(r_min, r_max, points):
        try:
            w = _solve(lambda w: w @ inp.cov @ w, n, allow_short,
                       [{"type": "eq", "fun": lambda w, t=target: w @ inp.mu - t}])
        except RuntimeError:
            continue
        s = port_stats(w, inp)
        rows.append({"vol": s["vol"], "ret": s["ret"], **dict(zip(inp.names, w))})
    return pd.DataFrame(rows)


def opportunity_set(inp: Inputs, samples: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Nuage de portefeuilles aléatoires (sans vente à découvert) : l'ensemble des possibles."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(inp.mu)), samples)
    ret = w @ inp.mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, inp.cov, w))
    return pd.DataFrame({"vol": vol, "ret": ret})


# ---------------------------------------------------------------------------
# 3. INVESTISSEUR : AVERSION AU RISQUE
# ---------------------------------------------------------------------------

MAX_LEVERAGE = 2.0  # avec levier, au plus 200 % investi dans le portefeuille risqué


def complete_portfolio(w_tan: np.ndarray, inp: Inputs, rf: float, A: float,
                       allow_leverage: bool = False) -> dict:
    """
    Part y* investie dans le portefeuille tangent, le reste au taux sans risque :
        y* = (E(r_T) - r_f) / (A * sigma_T^2)
    Plafonnée à 100 % sans levier, à MAX_LEVERAGE avec levier.
    """
    t = port_stats(w_tan, inp, rf)
    y_raw = (t["ret"] - rf) / (A * t["vol"] ** 2) if t["vol"] > 0 else 0.0
    y = max(0.0, min(MAX_LEVERAGE if allow_leverage else 1.0, y_raw))
    ret = rf + y * (t["ret"] - rf)
    vol = y * t["vol"]
    return {"y": y, "y_raw": y_raw, "ret": ret, "vol": vol,
            "utility": ret - 0.5 * A * vol ** 2,
            "weights": dict(zip(inp.names, y * w_tan)), "cash": 1 - y}
