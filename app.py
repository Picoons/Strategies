"""
Vitrine publique de stratégies d'investissement — Streamlit + yfinance

Lancer en local :   streamlit run app.py

Le site lit uniquement les fichiers JSON du dossier data/ (générés par export_public.py).
Ils ne contiennent aucun montant ni aucune quantité : seulement les poids en %
et les prix d'exécution des ordres.
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
}

CASH_LABEL = "Liquidités"

CLASS_COLORS = {
    "Or": "#d4a017",
    "Asie": "#dc2626",
    "Énergie": "#2563eb",
    CASH_LABEL: "#94a3b8",
}
PALETTE = ["#2563eb", "#d4a017", "#dc2626", "#16a34a", "#9333ea", "#0891b2",
           "#db2777", "#65a30d", "#ea580c", "#475569"]
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


def shade(hex_color: str, k: int) -> str:
    """k-ième nuance (plus claire) d'une couleur : 0 = couleur d'origine."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = min(0.2 * k, 0.6)
    r, g, b = (round(c + (255 - c) * f) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def line_colors(weights: pd.DataFrame) -> list[str]:
    """Chaque ligne prend une nuance de la couleur de sa classe."""
    classes = list(dict.fromkeys(weights["class"]))
    seen: dict[str, int] = {}
    out = []
    for c in weights["class"]:
        k = seen.get(c, 0)
        seen[c] = k + 1
        out.append(shade(class_color(c, classes.index(c)), k))
    return out


@st.cache_data(ttl=600)
def load_strategies() -> tuple[dict, list]:
    strategies, errors = {}, []
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            strategies[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{f.name} : {e}")
    return strategies, errors


@st.cache_data(ttl=3600, show_spinner="Récupération des cours sur Yahoo Finance…")
def get_prices(tickers: tuple, start: str) -> pd.DataFrame:
    return pf.fetch_prices_eur(list(tickers), pd.Timestamp(start))


def live_weights(strat: dict, last_px: pd.Series) -> pd.DataFrame:
    """
    Poids du jour de l'export, mis à jour avec l'évolution des cours depuis.
    """
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
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      separators=", ", legend=dict(font=dict(size=12)))
    return fig


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


def chart_orders_on_price(ticker: str, name: str, prices: pd.DataFrame, orders: pd.DataFrame) -> go.Figure:
    px = prices[ticker].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px.index, y=px, name=name, line=dict(color="#475569", width=1.6),
                             hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} €<extra></extra>"))
    for side, color, symbol, label in (("buy", C_POS, "triangle-up", "Achat"),
                                       ("sell", C_NEG, "triangle-down", "Vente")):
        o = orders[(orders["ticker"] == ticker) & (orders["side"] == side)]
        if not o.empty:
            fig.add_trace(go.Scatter(
                x=o["date"], y=o["price"], mode="markers", name=label,
                marker=dict(color=color, size=13, symbol=symbol, line=dict(color="white", width=1)),
                hovertemplate=f"{label} le %{{x|%d/%m/%Y}}<br>à %{{y:.2f}} €<extra></extra>",
            ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      separators=", ", yaxis=dict(ticksuffix=" €"))
    return fig


# ---------------------------------------------------------------------------
# PAGE D'UNE STRATÉGIE
# ---------------------------------------------------------------------------

def render_strategy(key: str, strat: dict, prices: pd.DataFrame):
    last_px = prices.ffill().iloc[-1]
    weights = live_weights(strat, last_px)
    orders = pd.DataFrame(strat["orders"])
    orders["date"] = pd.to_datetime(orders["date"])

    if key in DESCRIPTIONS:
        st.markdown(DESCRIPTIONS[key])

    c = st.columns(3)
    c[0].metric("Lancement", f"{pd.Timestamp(strat['start']):%d/%m/%Y}")
    c[1].metric("Ordres passés", len(orders))
    c[2].metric("Lignes en portefeuille", len(strat["positions"]))

    # --- Répartition
    st.subheader("Répartition actuelle")
    by_class = weights.groupby("class", sort=False)["weight"].sum().sort_values(ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Par classe d'actif")
        st.plotly_chart(
            chart_donut(by_class.index.tolist(), by_class.values.tolist(),
                        [class_color(l, i) for i, l in enumerate(by_class.index)]),
            use_container_width=True, key=f"cls_{key}")
    with col2:
        st.caption("Par ligne")
        st.plotly_chart(
            chart_donut(weights["name"].tolist(), weights["weight"].tolist(),
                        line_colors(weights)),
            use_container_width=True, key=f"line_{key}")

    target = TARGET_ALLOCATION.get(key)
    if target:
        st.caption("Allocation actuelle vs cible")
        st.plotly_chart(chart_target(by_class, target), use_container_width=True, key=f"tgt_{key}")

    # --- Ordres sur le graphique de cours
    st.subheader("Ordres sur le graphique")
    traded = orders.drop_duplicates("ticker")[["ticker", "name"]]
    options = dict(zip(traded["name"], traded["ticker"]))
    choice = st.selectbox("Actif", list(options.keys()), key=f"sel_{key}")
    if options[choice] in prices.columns:
        st.plotly_chart(chart_orders_on_price(options[choice], choice, prices, orders),
                        use_container_width=True, key=f"chart_{key}")

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
        use_container_width=True,
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
        with col:
            st.markdown(f"**{key}**")
            st.caption(f"Depuis le {pd.Timestamp(strat['start']):%d/%m/%Y} · {len(strat['orders'])} ordres")
            st.plotly_chart(
                chart_donut(by_class.index.tolist(), by_class.values.tolist(),
                            [class_color(l, i) for i, l in enumerate(by_class.index)]),
                use_container_width=True, key=f"ov_{key}")


# ---------------------------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------------------------

def main():
    st.title(f"📈 {SITE_TITLE}")

    strategies, errors = load_strategies()
    for e in errors:
        st.error(e)
    if not strategies:
        st.info("Aucune stratégie publiée. Lance `python export_public.py` pour générer les fichiers de data/.")
        st.stop()

    tickers = set()
    for s in strategies.values():
        tickers |= {p["ticker"] for p in s["positions"]} | {o["ticker"] for o in s["orders"]}
    start = min(pd.Timestamp(s["start"]) for s in strategies.values())

    try:
        prices = get_prices(tuple(sorted(tickers)), str(start.date()))
    except Exception as e:
        st.error(f"Impossible de récupérer les cours : {e}")
        st.stop()

    last = prices.dropna(how="all").index.max()
    st.caption(f"Cours au {last:%d/%m/%Y} · données Yahoo Finance (différées) · "
               "répartition en % de la valeur du portefeuille")

    if len(strategies) == 1:
        key, strat = next(iter(strategies.items()))
        st.header(key)
        render_strategy(key, strat, prices)
    else:
        tabs = st.tabs(["Vue d'ensemble"] + list(strategies.keys()))
        with tabs[0]:
            render_overview(strategies, prices)
        for tab, (key, strat) in zip(tabs[1:], strategies.items()):
            with tab:
                render_strategy(key, strat, prices)

    st.divider()
    st.caption("Stratégies personnelles présentées à titre informatif. "
               "Ceci ne constitue pas un conseil en investissement.")


main()
