"""
Vitrine publique de stratégies d'investissement — Streamlit + yfinance

Lancer en local :   streamlit run app.py

Le site lit uniquement les fichiers JSON du dossier data/ (générés par export_public.py).
Ils ne contiennent aucun montant ni aucune quantité : seulement les poids en %,
les prix d'exécution des ordres, les niveaux des trades et la performance en %.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import importlib

import portfolio as pf

# Recharge portfolio.py à chaque exécution : évite qu'une ancienne version reste en mémoire
# sur Streamlit Cloud après une mise à jour du fichier.
pf = importlib.reload(pf)

# ---------------------------------------------------------------------------
# CONFIGURATION DU SITE
# ---------------------------------------------------------------------------

SITE_TITLE = "Mes stratégies d'investissement"
DATA_DIR = Path(__file__).parent / "data"
REFLEXIONS_DIR = Path(__file__).parent / "reflexions"

# Dépôt GitHub du site : sert à lire les dates de publication / modification des réflexions.
# Détecté automatiquement quand c'est possible ; sinon, mets ici "ton-compte/ton-depot".
GITHUB_REPO = "Picoons/strategies"

# Allocation cible par stratégie (clé = nom du fichier sans extension)
TARGET_ALLOCATION = {
    "Donald - TR": {"Or": 1 / 3, "Asie": 1 / 3, "Énergie": 1 / 3},
}

# Nom affiché sur le site (clé = nom du fichier sans extension).
# Sans entrée ici, on affiche le début du nom de fichier ("Donald - TR" -> "Donald").
STRATEGY_NAMES = {
    "Donald - TR": "Macro - Long only strategy",
    "Daisy - TR": "Momentum - Long only strategy",
    "Picsou ETH - TR": "Crypto - Long/Short strategy",
}

# Texte de présentation par stratégie (facultatif)
DESCRIPTIONS = {
    "Donald - TR": "Portefeuille d'ETF équipondéré en trois blocs : "
                   "or physique, actions asiatiques et énergie mondiale.",
    "Daisy - TR": "Actions du CAC 40 sélectionnées par un screening hebdomadaire "
                  "basé sur l'indicateur Daisy.",
    "Picsou ETH - TR": "Stratégie systématique long / short sur Ethereum, "
                       "100 % du capital engagé sur chaque trade, sans levier.",
}

# Benchmarks affichés sur la courbe de performance (nom -> ticker Yahoo).
# Ils sont convertis dans la devise de la stratégie (€ pour Donald et Daisy, $ pour Picsou).
BENCHMARKS = {
    "Donald - TR": {"S&P 500": "^GSPC", "CAC 40": "^FCHI"},
    "Daisy - TR": {"CAC 40": "^FCHI"},
    "Picsou ETH - TR": {"Ethereum": "ETH-USD"},
}

CASH_LABEL = "Liquidités"
STRATEGY_OPTION = "📈 Stratégie"
SYMBOLS = {"EUR": "€", "USD": "$"}

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
    "Crypto": "#6366f1",
    CASH_LABEL: "#94a3b8",
}
PALETTE = ["#2563eb", "#d4a017", "#dc2626", "#16a34a", "#9333ea", "#0891b2",
           "#db2777", "#65a30d", "#ea580c", "#475569"]
BENCH_COLORS = ["#f59e0b", "#94a3b8", "#16a34a", "#9333ea"]
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


def display_name(key: str) -> str:
    return STRATEGY_NAMES.get(key, key.split(" - ")[0].strip())


def slug(text: str) -> str:
    """Texte -> identifiant d'URL : "Crypto - Long/Short strategy" -> "crypto-long-short-strategy"."""
    return re.sub(r"[^a-z0-9]+", "-", pf.normalize(text)).strip("-")


def currency(strat: dict) -> str:
    return strat.get("currency", "EUR")


def is_trades(strat: dict) -> bool:
    return strat.get("type") == "trades"


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
def get_prices(tickers: tuple, start: str, ccy: str, signature: tuple = ()) -> pd.DataFrame:
    return pf.fetch_prices(list(tickers), pd.Timestamp(start), ccy)


def strategy_tickers(key: str, strat: dict) -> set:
    if is_trades(strat):
        t = {strat["asset"]["ticker"]}
    else:
        t = {p["ticker"] for p in strat["positions"]} | {o["ticker"] for o in strat["orders"]}
    return t | set(BENCHMARKS.get(key, {}).values())


def trades_df(strat: dict) -> pd.DataFrame:
    df = pd.DataFrame(strat["trades"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["exit_date"] = df["exit_date"].astype(object).where(df["exit_date"].notna(), None)
    return df


def periods_per_year(strat: dict) -> int:
    """365 pour la crypto (cotée tous les jours), 252 sinon."""
    if is_trades(strat) and "-" in strat["asset"]["ticker"]:
        return 365
    return 252


# ---------------------------------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------------------------------

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


def portfolio_perf(strat: dict, prices: pd.DataFrame) -> pd.Series | None:
    """
    Stratégie de portefeuille : courbe calculée par export_public.py jusqu'à la date d'export,
    puis prolongée avec les cours du jour et les poids de l'export.
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


def strategy_perf(strat: dict, prices: pd.DataFrame) -> pd.Series | None:
    if is_trades(strat):
        t = strat["asset"]["ticker"]
        if t not in prices.columns or prices[t].isna().all():
            return None
        return pf.trades_perf(trades_df(strat), prices[t])
    return portfolio_perf(strat, prices)


def benchmark_perfs(key: str, prices: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
    """Performance de chaque benchmark sur la même période que la stratégie."""
    out = {}
    for name, t in BENCHMARKS.get(key, {}).items():
        if t not in prices.columns or prices[t].isna().all():
            continue
        px = prices[t].dropna()
        s = px.reindex(px.index.union(index)).sort_index().ffill().bfill().reindex(index)
        out[name] = s / s.iloc[0] - 1
    return out


def perf_kpis(perf: pd.Series, ppy: int = 252) -> dict:
    """
    Rendement total, rendement annualisé (géométrique), volatilité annualisée, drawdown max.
    Le rendement annualisé n'est affiché qu'à partir d'un an d'historique.
    """
    idx = 1 + perf
    idx = idx.asfreq("B").ffill() if ppy == 252 else idx.asfreq("D").ffill()
    total = float(idx.iloc[-1] / idx.iloc[0] - 1)
    years = (idx.index[-1] - idx.index[0]).days / 365.25
    annual = (1 + total) ** (1 / years) - 1 if years >= 1 else np.nan
    rets = idx.pct_change().dropna()
    if ppy == 252:
        rets = rets[rets != 0]
    vol = float(rets.std() * np.sqrt(ppy)) if len(rets) > 20 else np.nan
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


def chart_strategy(perf: pd.Series, name: str, benches: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf.index, y=perf, name=name, line=dict(color=C_LINE, width=2.2),
        fill="tozeroy", fillcolor="rgba(15,23,42,0.05)",
        hovertemplate="%{y:+.1%}",
    ))
    for i, (bname, s) in enumerate(benches.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=bname,
            line=dict(color=BENCH_COLORS[i % len(BENCH_COLORS)], width=1.6, dash="dot"),
            hovertemplate="%{y:+.1%}",
        ))
    return _pct_layout(fig, legend=bool(benches))


def chart_asset(ticker: str, name: str, prices: pd.DataFrame, start: pd.Timestamp) -> go.Figure:
    px = prices[ticker].dropna()
    px = px[px.index >= px.index[px.index <= start].max()] if (px.index <= start).any() else px
    evo = px / float(px.iloc[0]) - 1
    fig = go.Figure(go.Scatter(x=evo.index, y=evo, name=name,
                               line=dict(color="#475569", width=1.6),
                               hovertemplate="%{y:+.1%}"))
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
# BLOCS COMMUNS
# ---------------------------------------------------------------------------

def exposure(strat: dict, prices: pd.DataFrame) -> tuple[list, list, list]:
    """Libellés, poids et couleurs du camembert de répartition."""
    if is_trades(strat):
        open_t = [t for t in strat["trades"] if t["exit_date"] is None]
        if open_t:
            side = open_t[0]["side"]
            return ([f"{strat['asset']['name']} ({'long' if side == 'long' else 'short'})"], [1.0],
                    [C_POS if side == "long" else C_NEG])
        return [CASH_LABEL], [1.0], [CLASS_COLORS[CASH_LABEL]]
    w = live_weights(strat, prices.ffill().iloc[-1])
    by_class = w.groupby("class", sort=False)["weight"].sum().sort_values(ascending=False)
    return (by_class.index.tolist(), by_class.values.tolist(),
            [class_color(l, i) for i, l in enumerate(by_class.index)])


def render_top(key: str, strat: dict, prices: pd.DataFrame, perf: pd.Series | None,
               benches: dict, assets: dict, donut_title: str):
    """Camembert à gauche, évolution en % à droite."""
    labels, values, colors = exposure(strat, prices)
    left, right = st.columns(2)
    with left:
        st.subheader(donut_title)
        st.plotly_chart(chart_donut(labels, values, colors), width="stretch", key=f"donut_{key}")
    with right:
        st.subheader("Évolution")
        options = ([STRATEGY_OPTION] if perf is not None else []) + list(assets.keys())
        choice = st.selectbox("Afficher", options, key=f"sel_{key}", label_visibility="collapsed")
        if choice == STRATEGY_OPTION:
            st.plotly_chart(chart_strategy(perf, display_name(key), benches),
                            width="stretch", key=f"perf_{key}")
        elif assets.get(choice) in prices.columns:
            st.plotly_chart(chart_asset(assets[choice], choice, prices, pd.Timestamp(strat["start"])),
                            width="stretch", key=f"asset_{key}")
            st.caption("Évolution du cours depuis le lancement de la stratégie.")


def render_kpis(strat: dict, perf: pd.Series | None, benches: dict):
    if perf is None:
        st.info("Relance `python export_public.py` pour publier la courbe de performance.")
        return
    ppy = periods_per_year(strat)
    k = perf_kpis(perf, ppy)
    st.subheader("Performance")
    c = st.columns(4)
    c[0].metric("Rendement total", pct(k["total"]),
                help="Performance cumulée depuis le lancement, pondérée par le temps "
                     "(les apports d'argent ne la faussent pas).")
    c[1].metric("Rendement annualisé", pct(k["annual"]),
                help="Rendement moyen par an. Affiché à partir d'un an d'historique.")
    c[2].metric("Volatilité annualisée", pct(k["vol"], sign=False),
                help=f"Écart-type des rendements quotidiens × √{ppy}.")
    c[3].metric("Drawdown max", pct(k["max_dd"], sign=False),
                help="Plus forte baisse depuis un plus haut.")

    if benches:
        c = st.columns(4)
        for i, (name, s) in enumerate(benches.items()):
            b = perf_kpis(s, ppy)
            diff = (k["total"] - b["total"]) * 100
            c[i % 4].metric(f"{name} sur la même période", pct(b["total"]),
                            delta=f"{diff:+.1f} pts pour la stratégie".replace(".", ","),
                            help=f"Drawdown max de l'indice : {pct(b['max_dd'], sign=False)}")


# ---------------------------------------------------------------------------
# PAGE : STRATÉGIE DE PORTEFEUILLE (Donald, Daisy…)
# ---------------------------------------------------------------------------

def render_portfolio_strategy(key: str, strat: dict, prices: pd.DataFrame):
    sym = SYMBOLS.get(currency(strat), currency(strat))
    last_px = prices.ffill().iloc[-1]
    orders = pd.DataFrame(strat["orders"])
    orders["date"] = pd.to_datetime(orders["date"])
    perf = strategy_perf(strat, prices)
    benches = benchmark_perfs(key, prices, perf.index) if perf is not None else {}

    if key in DESCRIPTIONS:
        st.markdown(DESCRIPTIONS[key])
    c = st.columns(3)
    c[0].metric("Lancement", f"{pd.Timestamp(strat['start']):%d/%m/%Y}")
    c[1].metric("Ordres passés", len(orders))
    c[2].metric("Lignes en portefeuille", len(strat["positions"]))

    traded = orders.drop_duplicates("ticker")[["ticker", "name"]]
    render_top(key, strat, prices, perf, benches, dict(zip(traded["name"], traded["ticker"])),
               "Répartition actuelle")
    render_kpis(strat, perf, benches)

    target = TARGET_ALLOCATION.get(key)
    if target:
        w = live_weights(strat, last_px)
        by_class = w.groupby("class", sort=False)["weight"].sum()
        st.subheader("Allocation actuelle vs cible")
        st.plotly_chart(chart_target(by_class, target), width="stretch", key=f"tgt_{key}")

    st.subheader("Historique des ordres")
    table = orders.copy()
    table["Cours actuel"] = table["ticker"].map(last_px)
    table["Variation depuis l'ordre"] = (table["Cours actuel"] / table["price"] - 1) * 100
    table["Sens"] = table["side"].map({"buy": "🟢 Achat", "sell": "🔴 Vente"})
    table = table.rename(columns={"date": "Date", "name": "Actif", "class": "Classe",
                                  "ticker": "Ticker", "price": "Prix d'exécution"})
    table = table[["Date", "Sens", "Actif", "Classe", "Ticker", "Prix d'exécution",
                   "Cours actuel", "Variation depuis l'ordre"]].sort_values("Date", ascending=False)
    show_table(table, "Variation depuis l'ordre", {
        "Date": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Prix d'exécution": st.column_config.NumberColumn(format=f"%.2f {sym}"),
        "Cours actuel": st.column_config.NumberColumn(format=f"%.2f {sym}"),
        "Variation depuis l'ordre": st.column_config.NumberColumn(format="%+.1f %%"),
    })
    st.caption("Variation depuis l'ordre = cours actuel / prix d'exécution − 1. "
               "Pour une vente, une variation négative signifie que la sortie était bien placée.")


# ---------------------------------------------------------------------------
# PAGE : STRATÉGIE DE TRADING (Picsou…)
# ---------------------------------------------------------------------------

def render_trades_strategy(key: str, strat: dict, prices: pd.DataFrame):
    sym = SYMBOLS.get(currency(strat), currency(strat))
    asset = strat["asset"]
    trades = trades_df(strat)
    last = float(prices[asset["ticker"]].dropna().iloc[-1]) if asset["ticker"] in prices.columns else np.nan
    perf = strategy_perf(strat, prices)
    benches = benchmark_perfs(key, prices, perf.index) if perf is not None else {}

    # Résultat de chaque trade (en cours : au dernier cours)
    closed = trades["exit_date"].notna()
    trades["result"] = [
        pf.trade_return(t["side"], t["entry_price"], t["exit_price"] if c else last)
        for (_, t), c in zip(trades.iterrows(), closed)
    ]
    wins = int((trades.loc[closed, "result"] > 0).sum())
    n_closed = int(closed.sum())

    if key in DESCRIPTIONS:
        st.markdown(DESCRIPTIONS[key])
    c = st.columns(3)
    c[0].metric("Lancement", f"{pd.Timestamp(strat['start']):%d/%m/%Y}")
    c[1].metric("Trades", len(trades), help=f"{n_closed} clôturés, {len(trades) - n_closed} en cours")
    c[2].metric("Taux de réussite", pct(wins / n_closed, sign=False, decimals=0) if n_closed else "—",
                help=f"{wins} gagnants sur {n_closed} trades clôturés")

    render_top(key, strat, prices, perf, benches, {asset["name"]: asset["ticker"]}, "Position actuelle")
    render_kpis(strat, perf, benches)

    st.subheader("Historique des trades")
    today = pd.Timestamp.today().normalize()
    table = pd.DataFrame({
        "Entrée": trades["entry_date"],
        "Sortie": [pd.Timestamp(d).strftime("%d/%m/%Y") if c else "En cours"
                   for d, c in zip(trades["exit_date"], closed)],
        "Sens": trades["side"].map({"long": "🟢 Long", "short": "🔴 Short"}),
        "Prix d'entrée": trades["entry_price"],
        "Prix de sortie": [t["exit_price"] if c else last for (_, t), c in zip(trades.iterrows(), closed)],
        "Résultat": trades["result"] * 100,
        "Durée (jours)": [((pd.Timestamp(t["exit_date"]) if c else today) - t["entry_date"]).days
                          for (_, t), c in zip(trades.iterrows(), closed)],
    }).sort_values("Entrée", ascending=False)
    show_table(table, "Résultat", {
        "Entrée": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Prix d'entrée": st.column_config.NumberColumn(format=f"%.2f {sym}"),
        "Prix de sortie": st.column_config.NumberColumn(format=f"%.2f {sym}"),
        "Résultat": st.column_config.NumberColumn(format="%+.2f %%"),
    })
    st.caption("Pour un trade en cours, le prix de sortie est le dernier cours connu. "
               "Résultat d'un short = 1 − prix de sortie / prix d'entrée.")


def show_table(table: pd.DataFrame, color_col: str, config: dict):
    def color_var(v):
        if isinstance(v, (int, float)) and not np.isnan(v):
            return f"color: {C_POS}" if v > 0 else f"color: {C_NEG}"
        return ""

    st.dataframe(
        table.style.map(color_var, subset=[color_col]),
        hide_index=True,
        width="stretch",
        height=min(38 + 35 * len(table), 800),
        column_config=config,
    )


# ---------------------------------------------------------------------------
# PAGE : RÉFLEXIONS
# ---------------------------------------------------------------------------
# Chaque réflexion est un fichier Markdown (.md) dans le dossier reflexions/.
# En-tête facultatif en début de fichier :
#   ---
#   titre: Pourquoi l'or dans un portefeuille permanent
#   resume: Une phrase affichée dans la liste des réflexions.
#   date: 2026-09-29          (date de publication, si tu veux la forcer)
#   ---
# Sans en-tête : titre = première ligne "# ...", sinon le nom du fichier.
# Les dates de publication et de modification viennent de l'historique GitHub du fichier.

def parse_markdown(text: str) -> tuple[dict, str]:
    meta = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, flags=re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[pf.normalize(k)] = v.strip()
        text = text[m.end():]
    return meta, text.strip()


def detect_repo() -> str:
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"],
                             cwd=Path(__file__).parent, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        if m:
            return m.group(1)
    except Exception:
        pass
    return GITHUB_REPO


@st.cache_data(ttl=3600, show_spinner=False)
def github_dates(repo: str, path: str, signature: tuple) -> tuple[str | None, str | None]:
    """(première publication, dernière modification) d'un fichier, d'après ses commits GitHub."""
    try:
        url = f"https://api.github.com/repos/{repo}/commits?path={urllib.request.quote(path)}&per_page=100"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "streamlit-reflexions"})
        with urllib.request.urlopen(req, timeout=10) as r:
            commits = json.loads(r.read().decode())
        dates = [c["commit"]["committer"]["date"] for c in commits if "commit" in c]
        if dates:
            return min(dates), max(dates)
    except Exception:
        pass
    return None, None


def git_dates(path: Path) -> tuple[str | None, str | None]:
    """Repli : historique git local (si disponible)."""
    try:
        out = subprocess.run(["git", "log", "--follow", "--format=%cI", "--", str(path)],
                             cwd=Path(__file__).parent, capture_output=True, text=True,
                             timeout=5).stdout.split()
        if out:
            return out[-1], out[0]
    except Exception:
        pass
    return None, None


def reflexions_signature() -> tuple:
    if not REFLEXIONS_DIR.exists():
        return ()
    return tuple((f.name, f.stat().st_mtime, f.stat().st_size) for f in sorted(REFLEXIONS_DIR.glob("*.md")))


def load_reflexions() -> list[dict]:
    sig = reflexions_signature()
    repo = detect_repo()
    items = []
    for f in sorted(REFLEXIONS_DIR.glob("*.md")) if REFLEXIONS_DIR.exists() else []:
        meta, body = parse_markdown(f.read_text(encoding="utf-8"))
        title = meta.get("titre") or meta.get("title")
        if not title:
            m = re.match(r"^#\s+(.+)$", body, flags=re.M)
            if m and body.startswith("#"):
                title = m.group(1).strip()
                body = body[m.end():].strip()
        title = title or f.stem.replace("-", " ").replace("_", " ").capitalize()

        created, modified = github_dates(repo, f"reflexions/{f.name}", sig)
        if not created:
            created, modified = git_dates(f)
        if not modified:
            modified = pd.Timestamp(f.stat().st_mtime, unit="s").isoformat()
            created = created or modified
        if meta.get("date"):
            created = meta["date"]

        items.append({
            "slug": slug(f.stem),
            "title": title,
            "summary": meta.get("resume") or meta.get("summary") or "",
            "body": body,
            "created": pd.Timestamp(created).tz_localize(None) if pd.Timestamp(created).tzinfo is None
                       else pd.Timestamp(created).tz_convert("Europe/Paris").tz_localize(None),
            "modified": pd.Timestamp(modified).tz_localize(None) if pd.Timestamp(modified).tzinfo is None
                        else pd.Timestamp(modified).tz_convert("Europe/Paris").tz_localize(None),
        })
    return sorted(items, key=lambda x: x["modified"], reverse=True)


def dates_line(item: dict) -> str:
    line = f"Publié le {item['created']:%d/%m/%Y}"
    if item["modified"].date() > item["created"].date():
        line += f" · modifié le {item['modified']:%d/%m/%Y}"
    return line


def page_reflexions():
    items = load_reflexions()
    wanted = st.query_params.get("article", "")
    current = next((it for it in items if it["slug"] == wanted), None)

    if current:
        if st.button("← Toutes les réflexions"):
            del st.query_params["article"]
            st.rerun()
        st.title(current["title"])
        st.caption(dates_line(current))
        st.markdown(current["body"])
        return

    st.title("Réflexions")
    if not items:
        st.info("Aucune réflexion publiée pour l'instant.")
        return
    for it in items:
        with st.container(border=True):
            st.markdown(f"### {it['title']}")
            st.caption(dates_line(it))
            if it["summary"]:
                st.markdown(it["summary"])
            if st.button("Lire", key=f"read_{it['slug']}"):
                st.query_params["article"] = it["slug"]
                st.rerun()


# ---------------------------------------------------------------------------
# PAGE : STRATÉGIES
# ---------------------------------------------------------------------------

def load_market():
    """Stratégies publiées + cours (regroupés par devise). None si rien à afficher."""
    sig = data_signature()
    strategies, errors = load_strategies(sig)
    for e in errors:
        st.error(e)
    if not strategies:
        st.info("Aucune stratégie publiée. Lance `python export_public.py` pour générer les fichiers de data/.")
        return None

    # Cours regroupés par devise (EUR pour Donald et Daisy, USD pour Picsou…)
    by_ccy: dict[str, set] = {}
    starts: dict[str, pd.Timestamp] = {}
    for key, s in strategies.items():
        ccy = currency(s)
        by_ccy.setdefault(ccy, set()).update(strategy_tickers(key, s))
        starts[ccy] = min(starts.get(ccy, pd.Timestamp(s["start"])), pd.Timestamp(s["start"]))

    prices_by_ccy = {}
    try:
        for ccy, tickers in by_ccy.items():
            prices_by_ccy[ccy] = get_prices(tuple(sorted(tickers)), str(starts[ccy].date()), ccy, sig)
    except Exception as e:
        st.error(f"Impossible de récupérer les cours : {e}")
        return None
    return strategies, prices_by_ccy


def page_strategies():
    market = load_market()
    if market is None:
        return
    strategies, prices_by_ccy = market

    def prices_for(key: str) -> pd.DataFrame:
        return prices_by_ccy[currency(strategies[key])]

    # --- Menu de sélection de la stratégie
    # Lien partageable : ?strategie=permanent-portfolio (ou l'ancien nom : ?strategie=donald)
    keys = list(strategies.keys())
    names = {k: display_name(k) for k in keys}
    w = st.query_params.get("strategie", "").strip().lower()

    def matches(k: str) -> bool:
        return bool(w) and w in (slug(names[k]), slug(k.split(" - ")[0]),
                                 pf.normalize(k.split(" - ")[0]).split()[0])

    default = next((i for i, k in enumerate(keys) if matches(k)), 0)
    col, _ = st.columns([1, 3])
    choice = col.selectbox("Stratégie", keys, index=default,
                           format_func=lambda k: names.get(k, k))
    st.query_params["strategie"] = slug(names[choice])

    prices = prices_for(choice)
    strat = strategies[choice]
    if prices.dropna(how="all").empty:
        st.error("Yahoo Finance n'a renvoyé aucun cours. Réessaie dans quelques minutes.")
        return
    last = prices.dropna(how="all").index.max()
    st.title(names[choice])
    st.caption(f"Cours au {last:%d/%m/%Y} · données Yahoo Finance (différées) · "
               f"performance calculée en {currency(strat)}")
    if is_trades(strat):
        render_trades_strategy(choice, strat, prices)
    else:
        render_portfolio_strategy(choice, strat, prices)

    st.divider()
    st.caption("Stratégies personnelles présentées à titre informatif, hors frais et fiscalité. "
               "Les indices de référence sont hors dividendes. "
               "Ceci ne constitue pas un conseil en investissement.")


# ---------------------------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------------------------

SECTIONS = {"strategies": "📈 Stratégies", "reflexions": "✍️ Réflexions"}


def main():
    # Lien partageable : ?page=reflexions (ou ?page=strategies)
    current = st.query_params.get("page", "strategies")
    if current not in SECTIONS:
        current = "strategies"
    section = st.segmented_control("Section", list(SECTIONS.keys()), default=current,
                                   format_func=SECTIONS.get, label_visibility="collapsed",
                                   key="section")
    section = section or current  # un second clic sur l'onglet actif ne le désélectionne pas

    previous = st.query_params.get("page")
    if previous is None:
        st.query_params["page"] = section
    elif section != previous:
        # Changement de section : on repart d'une URL propre
        st.query_params.clear()
        st.query_params["page"] = section

    if section == "reflexions":
        page_reflexions()
    else:
        page_strategies()


main()
