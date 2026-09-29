"""
Vitrine publique de stratégies d'investissement — Streamlit + yfinance

Lancer en local :   streamlit run app.py

Le site lit uniquement les fichiers JSON du dossier data/ (générés par export_public.py).
Ils ne contiennent aucun montant ni aucune quantité : seulement les poids en %,
les prix d'exécution des ordres et la performance en %.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import portfolio as pf

# ---------------------------------------------------------------------------
# CONFIGURATION DU SITE
# ---------------------------------------------------------------------------

SITE_TITLE = "Mes stratégies d'investissement"
DATA_DIR = Path(__file__).parent / "data"

# Allocation cible par stratégie (clé = nom du fichier sans extension)
TARGET_ALLOCATION = {
    "Donald - TR": {"Or": 1 / 3, "Asie": 1 / 3, "Énergie": 1 / 3},
}

# Texte de présentation par stratégie (facultatif)
DESCRIPTIONS = {
    "Donald - TR": "Portefeuille d'ETF équipondéré en trois blocs : "
                   "or physique, actions asiatiques et énergie mondiale.",
    "Daisy - TR": "Actions du CAC 40 sélectionnées par un screening hebdomadaire "
                  "basé sur l'indicateur Daisy.",
}

COMPARE_OPTION = "Comparer les stratégies"

CASH_LABEL = "Liquidités"
STRATEGY_OPTION = "📈 Stratégie"

CLASS_COLORS = {
    "Or": "#d4a017",
    "Asie": "#dc2626",
    "Énergie": "#2563eb",
    "Industrie": "#0891b2",
    "Consommation de base": "#16a34a",
    "Consommation cyclique": "#65a30d",
    "Télécoms & médias": "#9333ea",
    "Technologie": "#4f46e5",
    "Finance": "#0f766e",
    "Santé": "#db2777",
    "Matériaux": "#a16207",
    "Services publics": "#ea580c",
    "Immobilier": "#78716c",
    CASH_LABEL: "#94a3b8",
}
PALETTE = ["#2563eb", "#d4a017", "#dc2626", "#16a34a", "#9333ea", "#0891b2",
           "#db2777", "#65a30d", "#ea580c", "#475569"]
C_LINE = "#0f172a"
C_POS = "#16a34a"
C_NEG = "#dc2626"

st.set_page_config(page_title=SITE_TITLE, page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# OUTILS
# ---------------------------------------------------------------------------

def pct(x, sign: bool = True, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x * 100:+.{decimals}f}" if sign else f"{x * 100:.{decimals}f}"
    return s.replace(".", ",") + " %"


def class_color(label: str, i: int) -> str:
    return CLASS_COLORS.get(label, PALETTE[i % len(PALETTE)])


def data_signature() -> tuple:
    """Change dès qu'un fichier de data/ est ajouté ou modifié -> rechargement immédiat."""
    return tuple((f.name, f.stat().st_mtime, f.stat().st_size) for f in sorted(DATA_DIR.glob("*.json")))


@st.cache_data
def load_strategies(signature: tuple) -> tuple[dict, list]:
    strategies, errors = {}, []
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            strategies[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{f.name} : {e}")
    return strategies, errors


@st.cache_data(ttl=3600, show_spinner="Récupération des cours sur Yahoo Finance…")
def get_prices(tickers: tuple, start: str, signature: tuple = ()) -> pd.DataFrame:
    return pf.fetch_prices_eur(list(tickers), pd.Timestamp(start))


def live_weights(strat: dict, last_px: pd.Series) -> pd.DataFrame:
    """Poids du jour de l'export, mis à jour avec l'évolution des cours depuis."""
    rows = []
    for p in strat["positions"]:
        now = last_px.get(p["ticker"], np.nan)
        drift = now / p["ref_price"] if p["ref_price"] and pd.notna(now) else 1.0
        rows.append({"name": p["name"], "class": p["class"], "ticker": p["ticker"],
                     "raw": p["weight"] * drift})
    if strat.get("cash_weight", 0) > 0.005:
        rows.append({"name": CASH_LABEL, "class": CASH_LABEL, "ticker": "",
                     "raw": strat["cash_weight"]})
    df = pd.DataFrame(rows)
    df["weight"] = df["raw"] / df["raw"].sum()
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def strategy_perf(strat: dict, prices: pd.DataFrame) -> pd.Series | None:
    """
    Performance cumulée de la stratégie (en décimal, 0.12 = +12 %).
    - jusqu'à la date d'export : courbe calculée par export_public.py
    - après : prolongée avec les cours du jour et les poids de l'export
    """
    perf = strat.get("perf")
    if not perf or not perf.get("dates"):
        return None
    hist = pd.Series(perf["twr"], index=pd.to_datetime(perf["dates"]))
    as_of = hist.index.max()

    px = prices.ffill()
    after = px.index[px.index > as_of]
    if len(after) == 0:
        return hist

    ratio = pd.Series(strat.get("cash_weight", 0.0), index=after)
    for p in strat["positions"]:
        t = p["ticker"]
        if t in px.columns and p["ref_price"]:
            ratio = ratio + p["weight"] * px.loc[after, t] / p["ref_price"]
        else:
            ratio = ratio + p["weight"]
    ratio = ratio / (sum(p["weight"] for p in strat["positions"]) + strat.get("cash_weight", 0.0))
    ext = (1 + hist.iloc[-1]) * ratio - 1
    return pd.concat([hist, ext])


def perf_kpis(perf: pd.Series) -> dict:
    """Rendement total, rendement annualisé, volatilité annualisée, drawdown max."""
    idx = (1 + perf).asfreq("B").ffill()
    total = float(idx.iloc[-1] / idx.iloc[0] - 1)
    years = (idx.index[-1] - idx.index[0]).days / 365.25
    annual = (1 + total) ** (1 / years) - 1 if years >= 1 else np.nan
    rets = idx.pct_change().dropna()
    rets = rets[rets != 0]
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 20 else np.nan
    dd = float((idx / idx.cummax() - 1).min())
    return {"total": total, "annual": annual, "vol": vol, "max_dd": dd, "years": years}


# ---------------------------------------------------------------------------
# GRAPHIQUES
# ---------------------------------------------------------------------------

def chart_donut(labels: list, values: list, colors: list) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6, sort=False,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        texttemplate="%{percent:.0%}",
        hovertemplate="%{label}<br>%{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10),
                      separators=", ", legend=dict(font=dict(size=13)))
    return fig


def _pct_layout(fig: go.Figure, legend: bool = False) -> go.Figure:
    ys = [v for tr in fig.data for v in (tr.y if tr.y is not None else [])]
    span = (max(ys) - min(ys)) if ys else 1
    fig.update_yaxes(tickformat="+.1%" if span < 0.06 else "+.0%")
    fig.update_xaxes(tickformat="%d/%m/%y")
    fig.add_hline(y=0, line=dict(color="#cbd5e1", width=1))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      separators=", ")
    return fig


def chart_strategy(perf: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf.index, y=perf, name=name, line=dict(color=C_LINE, width=2),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.05)",
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:+.1%}<extra></extra>",
    ))
    return _pct_layout(fig)


def chart_asset(ticker: str, name: str, prices: pd.DataFrame, start: pd.Timestamp) -> go.Figure:
    px = prices[ticker].dropna()
    px = px[px.index >= px.index[px.index <= start].max()] if (px.index <= start).any() else px
    base = float(px.iloc[0])
    evo = px / base - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=evo.index, y=evo, name=name, line=dict(color="#475569", width=1.6),
                             hovertemplate="%{x|%d/%m/%Y}<br>%{y:+.1%}<extra></extra>"))
    return _pct_layout(fig)


def chart_target(current: pd.Series, target: dict) -> go.Figure:
    classes = list(target.keys())
    cur = [float(current.get(c, 0.0)) for c in classes]
    tgt = [target[c] for c in classes]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=classes, x=tgt, orientation="h", name="Cible",
                         marker_color="#e2e8f0",
                         text=[pct(v, sign=False, decimals=0) for v in tgt], textposition="outside",
                         hovertemplate="Cible : %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(y=classes, x=cur, orientation="h", name="Actuelle",
                         marker_color=[class_color(c, i) for i, c in enumerate(classes)],
                         text=[pct(v, sign=False, decimals=0) for v in cur], textposition="outside",
                         hovertemplate="Actuelle : %{x:.1%}<extra></extra>"))
    fig.update_layout(barmode="group", height=280, margin=dict(l=10, r=40, t=30, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), separators=", ",
                      xaxis=dict(tickformat=".0%", range=[0, max(cur + tgt) * 1.3]),
                      yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# PAGE D'UNE STRATÉGIE
# ---------------------------------------------------------------------------

def render_strategy(key: str, strat: dict, prices: pd.DataFrame):
    last_px = prices.ffill().iloc[-1]
    weights = live_weights(strat, last_px)
    orders = pd.DataFrame(strat["orders"])
    orders["date"] = pd.to_datetime(orders["date"])
    start = pd.Timestamp(strat["start"])
    perf = strategy_perf(strat, prices)

    if key in DESCRIPTIONS:
        st.markdown(DESCRIPTIONS[key])

    c = st.columns(3)
    c[0].metric("Lancement", f"{start:%d/%m/%Y}")
    c[1].metric("Ordres passés", len(orders))
    c[2].metric("Lignes en portefeuille", len(strat["positions"]))

    # --- Répartition (gauche) + évolution en % (droite)
    by_class = weights.groupby("class", sort=False)["weight"].sum().sort_values(ascending=False)
    left, right = st.columns(2)
    with left:
        st.subheader("Répartition actuelle")
        st.plotly_chart(
            chart_donut(by_class.index.tolist(), by_class.values.tolist(),
                        [class_color(l, i) for i, l in enumerate(by_class.index)]),
            width="stretch", key=f"cls_{key}")
    with right:
        st.subheader("Évolution")
        traded = orders.drop_duplicates("ticker")[["ticker", "name"]]
        assets = dict(zip(traded["name"], traded["ticker"]))
        options = ([STRATEGY_OPTION] if perf is not None else []) + list(assets.keys())
        choice = st.selectbox("Afficher", options, key=f"sel_{key}", label_visibility="collapsed")
        if choice == STRATEGY_OPTION:
            st.plotly_chart(chart_strategy(perf, key), width="stretch",
                            key=f"perf_{key}")
        elif assets[choice] in prices.columns:
            st.plotly_chart(chart_asset(assets[choice], choice, prices, start),
                            width="stretch", key=f"asset_{key}")
            st.caption("Évolution du cours depuis le lancement de la stratégie.")

    # --- KPI de performance
    if perf is not None:
        k = perf_kpis(perf)
        st.subheader("Performance")
        c = st.columns(4)
        c[0].metric("Rendement total", pct(k["total"]),
                    help="Performance cumulée depuis le lancement, pondérée par le temps "
                         "(les apports d'argent ne la faussent pas).")
        c[1].metric("Rendement annualisé", pct(k["annual"]),
                    help="Rendement moyen par an. Affiché à partir d'un an d'historique.")
        c[2].metric("Volatilité annualisée", pct(k["vol"], sign=False),
                    help="Écart-type des rendements quotidiens × √252.")
        c[3].metric("Drawdown max", pct(k["max_dd"], sign=False),
                    help="Plus forte baisse depuis un plus haut.")
    else:
        st.info("Relance `python export_public.py` pour publier la courbe de performance.")

    # --- Cible
    target = TARGET_ALLOCATION.get(key)
    if target:
        st.subheader("Allocation actuelle vs cible")
        st.plotly_chart(chart_target(by_class, target), width="stretch", key=f"tgt_{key}")

    # --- Historique des ordres
    st.subheader("Historique des ordres")
    table = orders.copy()
    table["Cours actuel"] = table["ticker"].map(last_px)
    table["Variation depuis l'ordre"] = (table["Cours actuel"] / table["price"] - 1) * 100
    table["Sens"] = table["side"].map({"buy": "🟢 Achat", "sell": "🔴 Vente"})
    table = table.rename(columns={"date": "Date", "name": "Actif", "class": "Classe",
                                  "ticker": "Ticker", "price": "Prix d'exécution"})
    table = table[["Date", "Sens", "Actif", "Classe", "Ticker", "Prix d'exécution",
                   "Cours actuel", "Variation depuis l'ordre"]].sort_values("Date", ascending=False)

    def color_var(v):
        if isinstance(v, (int, float)) and not np.isnan(v):
            return f"color: {C_POS}" if v > 0 else f"color: {C_NEG}"
        return ""

    st.dataframe(
        table.style.map(color_var, subset=["Variation depuis l'ordre"]),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(table), 800),
        column_config={
            "Date": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Prix d'exécution": st.column_config.NumberColumn(format="%.2f €"),
            "Cours actuel": st.column_config.NumberColumn(format="%.2f €"),
            "Variation depuis l'ordre": st.column_config.NumberColumn(format="%+.1f %%"),
        },
    )
    st.caption("Variation depuis l'ordre = cours actuel / prix d'exécution − 1. "
               "Pour une vente, une variation négative signifie que la sortie était bien placée.")


def render_overview(strategies: dict, prices: pd.DataFrame):
    last_px = prices.ffill().iloc[-1]
    cols = st.columns(len(strategies))
    for col, (key, strat) in zip(cols, strategies.items()):
        w = live_weights(strat, last_px)
        by_class = w.groupby("class", sort=False)["weight"].sum().sort_values(ascending=False)
        perf = strategy_perf(strat, prices)
        with col:
            st.markdown(f"**{key}**")
            st.caption(f"Depuis le {pd.Timestamp(strat['start']):%d/%m/%Y} · {len(strat['orders'])} ordres")
            if perf is not None:
                st.metric("Rendement total", pct(perf_kpis(perf)["total"]))
            st.plotly_chart(
                chart_donut(by_class.index.tolist(), by_class.values.tolist(),
                            [class_color(l, i) for i, l in enumerate(by_class.index)]),
                width="stretch", key=f"ov_{key}")

    series = {k: strategy_perf(s, prices) for k, s in strategies.items()}
    series = {k: v for k, v in series.items() if v is not None}
    if series:
        fig = go.Figure()
        for i, (k, s) in enumerate(series.items()):
            fig.add_trace(go.Scatter(x=s.index, y=s, name=k,
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                     hovertemplate="%{x|%d/%m/%Y}<br>%{y:+.1%}<extra></extra>"))
        st.subheader("Évolution comparée")
        st.plotly_chart(_pct_layout(fig, legend=True), width="stretch", key="ov_perf")


# ---------------------------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------------------------

def main():
    strategies, errors = load_strategies(data_signature())
    for e in errors:
        st.error(e)
    if not strategies:
        st.title(f"📈 {SITE_TITLE}")
        st.info("Aucune stratégie publiée. Lance `python export_public.py` pour générer les fichiers de data/.")
        st.stop()

    tickers = set()
    for s in strategies.values():
        tickers |= {p["ticker"] for p in s["positions"]} | {o["ticker"] for o in s["orders"]}
    start = min(pd.Timestamp(s["start"]) for s in strategies.values())

    try:
        prices = get_prices(tuple(sorted(tickers)), str(start.date()), data_signature())
    except Exception as e:
        st.error(f"Impossible de récupérer les cours : {e}")
        st.stop()

    last = prices.dropna(how="all").index.max()

    # --- Menu de sélection de la stratégie (lien partageable : ?strategie=Daisy)
    keys = list(strategies.keys())
    options = keys + ([COMPARE_OPTION] if len(keys) > 1 else [])
    short = {k: k.split(" - ")[0].strip() for k in keys}
    wanted = st.query_params.get("strategie", "")
    default = next((i for i, k in enumerate(keys) if short[k].lower() == wanted.lower()), 0)
    if wanted.lower() == "comparaison" and len(keys) > 1:
        default = len(keys)
    col, _ = st.columns([1, 3])
    choice = col.selectbox("Stratégie", options, index=default,
                           format_func=lambda k: short.get(k, k))
    st.query_params["strategie"] = short.get(choice, "comparaison")

    if choice == COMPARE_OPTION:
        st.title("Comparaison des stratégies")
        st.caption(f"Cours au {last:%d/%m/%Y} · données Yahoo Finance (différées)")
        render_overview(strategies, prices)
    else:
        st.title(short[choice])
        st.caption(f"Cours au {last:%d/%m/%Y} · données Yahoo Finance (différées)")
        render_strategy(choice, strategies[choice], prices)

    st.divider()
    st.caption("Stratégies personnelles présentées à titre informatif, hors frais et fiscalité. "
               "Ceci ne constitue pas un conseil en investissement.")


main()
