"""
Moteur de calcul du suivi de portefeuille.

- Lit les fichiers Excel de transactions (une stratégie = un fichier)
- Récupère les prix via yfinance et les convertit en EUR
- Calcule positions, cash, valeur quotidienne, P&L et performance (TWR)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Correspondance "nom de l'actif dans l'Excel" -> ticker Yahoo Finance.
# Utilisée seulement si la colonne "Ticker" de l'Excel est vide.
# Toutes les cotations ci-dessous sont sur Xetra (.DE) ou Milan (.MI), donc déjà en EUR.
TICKER_MAP = {
    "Isahres - Physical gold USD (Acc)": "PPFB.DE",   # iShares Physical Gold ETC - IE00B4ND3602
    "MSCI AC Far es Ex Japan USD (Acc)": "IQQF.DE",   # iShares MSCI AC Far East ex-Japan - IE00B0M63730
    "MSCI World Energy USD (Acc)": "XDW0.DE",         # Xtrackers MSCI World Energy 1C - IE00BM67HM91
    "Physical Swiss gold USD": "SGBS.MI",             # WisdomTree Physical Swiss Gold - JE00B588CD74 (Milan, EUR)
    "MSCI China Tech USD (Acc)": "CBUK.DE",           # iShares MSCI China Tech Acc - IE000NFR7C63
    "Hang Seng Tech HKD (Acc)": "H4ZX.DE",            # HSBC Hang Seng TECH - IE00BMWXKN31
}

# Classe d'actif par ticker (pour le graphique de répartition).
ASSET_CLASS = {
    "PPFB.DE": "Or",
    "SGBS.MI": "Or",
    "IQQF.DE": "Asie",
    "CBUK.DE": "Asie",
    "H4ZX.DE": "Asie",
    "XDW0.DE": "Énergie",
}

# Nom affiché sur le site (sinon on garde le nom de l'Excel)
DISPLAY_NAME = {
    "PPFB.DE": "iShares Physical Gold",
    "SGBS.MI": "WisdomTree Physical Swiss Gold",
    "IQQF.DE": "iShares MSCI AC Far East ex-Japan",
    "CBUK.DE": "iShares MSCI China Tech",
    "H4ZX.DE": "HSBC Hang Seng TECH",
    "XDW0.DE": "Xtrackers MSCI World Energy",
}

# Réglages propres à chaque stratégie (clé = nom du fichier Excel sans extension)
#   unit_price : True si la colonne "P revient" contient le prix UNITAIRE (sinon = montant total)
#   suffix     : suffixe Yahoo ajouté aux tickers écrits sans place de cotation (DG -> DG.PA)
#   format="trades" : journal de trades transposé (une colonne par trade : Entry date,
#                     Entry level, Direction Long/Short, Exit date, Exit level)
STRATEGY_SETTINGS = {
    "Daisy - TR": {"unit_price": True, "suffix": ".PA"},
    "Picsou ETH - TR": {
        "format": "trades",
        "ticker": "ETH-USD",
        "name": "Ethereum",
        "class": "Crypto",
        "currency": "USD",
    },
}

# Traduction des secteurs Yahoo (utilisée quand un ticker n'est pas dans ASSET_CLASS)
SECTOR_FR = {
    "Industrials": "Industrie",
    "Consumer Defensive": "Consommation de base",
    "Consumer Cyclical": "Consommation cyclique",
    "Communication Services": "Télécoms & médias",
    "Technology": "Technologie",
    "Financial Services": "Finance",
    "Healthcare": "Santé",
    "Energy": "Énergie",
    "Basic Materials": "Matériaux",
    "Utilities": "Services publics",
    "Real Estate": "Immobilier",
}

# Benchmarks proposés dans l'interface (nom -> ticker Yahoo)
BENCHMARKS = {
    "MSCI World (EUNL.DE)": "EUNL.DE",
    "S&P 500 (SXR8.DE)": "SXR8.DE",
    "CAC 40 (^FCHI)": "^FCHI",
    "Or (PPFB.DE)": "PPFB.DE",
    "Bitcoin (BTC-EUR)": "BTC-EUR",
}

# Libellés reconnus dans l'Excel (comparaison sans accents ni majuscules)
DEPOSIT_WORDS = ("depot", "versement", "deposit")
WITHDRAW_WORDS = ("retrait", "withdrawal")
DIVIDEND_WORDS = ("dividende", "dividend", "coupon", "interet", "interets")
BUY_WORDS = ("achat", "buy", "achat programme", "savings plan")
SELL_WORDS = ("vente", "sell")

# Noms de colonnes possibles dans l'Excel
COLUMN_ALIASES = {
    "date": ("date",),
    "asset": ("asset", "actif", "nom", "produit"),
    "ticker": ("ticker", "symbole", "symbol"),
    "direction": ("direction", "sens", "type", "operation"),
    "qty": ("qtt", "qte", "quantite", "qty", "quantity", "nombre"),
    "amount": ("p revient", "prix de revient", "montant", "amount"),
}


# ---------------------------------------------------------------------------
# LECTURE DES TRANSACTIONS
# ---------------------------------------------------------------------------

def normalize(text) -> str:
    """Minuscules, sans accents, espaces nettoyés."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().strip().split())


def _starts_with_any(value: str, words) -> bool:
    return any(value.startswith(w) for w in words)


def _find_columns(df: pd.DataFrame) -> dict:
    found = {}
    norm_cols = {normalize(c): c for c in df.columns}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in norm_cols:
                found[key] = norm_cols[alias]
                break
    missing = [k for k in ("date", "asset", "amount") if k not in found]
    if missing:
        raise ValueError(
            f"Colonnes introuvables : {missing}. Colonnes lues : {list(df.columns)}"
        )
    return found


def _add_suffix(ticker: str, suffix: str) -> str:
    if ticker and suffix and not any(c in ticker for c in ".-^="):
        return ticker.upper() + suffix
    return ticker


def load_transactions(source, settings: dict | None = None,
                      ticker_map: dict | None = None) -> pd.DataFrame:
    """
    Lit un Excel de transactions et renvoie un DataFrame normalisé :
    date, asset, ticker, kind (deposit/withdraw/buy/sell/dividend), qty, amount (montant total)

    - Sans colonne "Direction" : quantité positive = achat, négative = vente.
    - settings["unit_price"] : la colonne "P revient" est un prix unitaire.
    - settings["suffix"] : suffixe ajouté aux tickers sans place de cotation.
    """
    settings = settings or {}
    unit_price = settings.get("unit_price", False)
    suffix = settings.get("suffix", "")
    ticker_map = ticker_map or TICKER_MAP
    norm_map = {normalize(k): v for k, v in ticker_map.items()}

    raw = pd.read_excel(source)
    cols = _find_columns(raw)

    rows = []
    for _, r in raw.iterrows():
        date = pd.to_datetime(r[cols["date"]], errors="coerce", dayfirst=True)
        if pd.isna(date):
            continue
        asset = str(r[cols["asset"]]).strip() if pd.notna(r[cols["asset"]]) else ""
        asset_n = normalize(asset)
        direction_n = normalize(r[cols["direction"]]) if "direction" in cols else ""
        amount = pd.to_numeric(r[cols["amount"]], errors="coerce")
        qty = pd.to_numeric(r[cols["qty"]], errors="coerce") if "qty" in cols else np.nan
        ticker = ""
        if "ticker" in cols and pd.notna(r[cols["ticker"]]):
            ticker = str(r[cols["ticker"]]).strip()

        if pd.isna(amount):
            continue
        amount = abs(float(amount))
        signed_qty = float(qty) if pd.notna(qty) else np.nan
        qty = abs(signed_qty) if pd.notna(signed_qty) else np.nan

        if _starts_with_any(asset_n, DEPOSIT_WORDS) or _starts_with_any(direction_n, DEPOSIT_WORDS):
            kind = "deposit"
        elif _starts_with_any(asset_n, WITHDRAW_WORDS) or _starts_with_any(direction_n, WITHDRAW_WORDS):
            kind = "withdraw"
        elif _starts_with_any(direction_n, DIVIDEND_WORDS) or _starts_with_any(asset_n, DIVIDEND_WORDS):
            kind = "dividend"
        elif _starts_with_any(direction_n, SELL_WORDS):
            kind = "sell"
        elif _starts_with_any(direction_n, BUY_WORDS):
            kind = "buy"
        elif not direction_n and pd.notna(signed_qty) and signed_qty != 0:
            kind = "buy" if signed_qty > 0 else "sell"
        else:
            raise ValueError(f"Ligne non reconnue ({date.date()} - {asset} - {direction_n!r})")

        if kind in ("buy", "sell"):
            if not ticker:
                ticker = norm_map.get(asset_n, "")
            ticker = _add_suffix(ticker, suffix)
            if pd.isna(qty) or qty <= 0:
                raise ValueError(f"Quantité manquante ({date.date()} - {asset})")
            if unit_price:
                amount = amount * qty

        rows.append(
            {
                "date": date.normalize(),
                "asset": asset,
                "ticker": ticker,
                "kind": kind,
                "qty": float(qty) if kind in ("buy", "sell") else 0.0,
                "amount": amount,
            }
        )

    tx = pd.DataFrame(rows)
    if tx.empty:
        raise ValueError("Aucune transaction lue dans le fichier.")
    # Ordre stable : date, puis dépôts avant achats avant ventes le même jour
    order = {"deposit": 0, "dividend": 1, "sell": 2, "buy": 3, "withdraw": 4}
    tx["_o"] = tx["kind"].map(order)
    tx = tx.sort_values(["date", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    return tx


# ---------------------------------------------------------------------------
# STRATÉGIES DE TRADING (journal de trades, sans quantités)
# ---------------------------------------------------------------------------

TRADE_ROWS = {
    "entry_date": ("entry date", "date d'entree", "date entree"),
    "entry_price": ("entry level", "entry price", "prix d'entree", "niveau d'entree"),
    "side": ("direction", "sens"),
    "exit_date": ("exit date", "date de sortie", "date sortie"),
    "exit_price": ("exit level", "exit price", "prix de sortie", "niveau de sortie"),
}


def load_trades(source) -> pd.DataFrame:
    """
    Lit un journal de trades transposé (une colonne par trade).
    Les lignes non reconnues (durée, P&L, winrate...) sont ignorées.
    Renvoie : entry_date, entry_price, side (long/short), exit_date, exit_price
    (exit_* vides si le trade est encore ouvert).
    """
    raw = pd.read_excel(source, header=None)
    labels = {normalize(v): i for i, v in raw.iloc[:, 0].items() if pd.notna(v)}
    rows = {}
    for key, aliases in TRADE_ROWS.items():
        idx = next((labels[a] for a in aliases if a in labels), None)
        if idx is None and key in ("entry_date", "entry_price", "side"):
            raise ValueError(f"Ligne '{aliases[0]}' introuvable dans le journal de trades.")
        rows[key] = idx

    trades = []
    for col in range(1, raw.shape[1]):
        def cell(key):
            i = rows.get(key)
            return raw.iat[i, col] if i is not None else None

        entry_date = pd.to_datetime(cell("entry_date"), errors="coerce")
        entry_price = pd.to_numeric(cell("entry_price"), errors="coerce")
        side = normalize(cell("side"))
        if pd.isna(entry_date) or pd.isna(entry_price) or side not in ("long", "short"):
            continue
        exit_date = pd.to_datetime(cell("exit_date"), errors="coerce")
        exit_price = pd.to_numeric(cell("exit_price"), errors="coerce")
        closed = pd.notna(exit_date) and pd.notna(exit_price)
        trades.append({
            "entry_date": entry_date.normalize(),
            "entry_price": float(entry_price),
            "side": side,
            "exit_date": exit_date.normalize() if closed else None,
            "exit_price": float(exit_price) if closed else None,
        })
    if not trades:
        raise ValueError("Aucun trade lu dans le fichier.")
    df = pd.DataFrame(trades).sort_values("entry_date", kind="stable").reset_index(drop=True)
    open_count = df["exit_date"].isna().sum()
    if open_count > 1:
        raise ValueError("Plusieurs trades ouverts en même temps : un seul est autorisé.")
    return df


def trade_return(side: str, entry: float, price: float) -> float:
    """Rendement d'un trade (en décimal) au prix `price`."""
    return price / entry - 1 if side == "long" else 1 - price / entry


def trades_perf(trades: pd.DataFrame, px: pd.Series, end=None) -> pd.Series:
    """
    Performance cumulée d'une stratégie de trading à 100 % du capital :
    - en position : suit le sous-jacent (long) ou son inverse (short), sans levier ;
    - hors position : capital inchangé ;
    - entrées et sorties aux niveaux indiqués dans le journal, clôtures quotidiennes entre les deux.
    """
    px = px.dropna()
    px.index = pd.to_datetime(px.index).normalize()
    end = pd.Timestamp(end or px.index.max()).normalize()
    start = pd.Timestamp(trades["entry_date"].min())
    grid = pd.date_range(start, end, freq="D")
    close = px.reindex(px.index.union(grid)).sort_index().ffill().bfill().reindex(grid)

    equity = pd.Series(np.nan, index=grid)
    eq = 1.0
    for _, t in trades.iterrows():
        e_d = pd.Timestamp(t["entry_date"])
        x_d = pd.Timestamp(t["exit_date"]) if t["exit_date"] is not None and pd.notna(t["exit_date"]) else None
        last_day = min(x_d, end) if x_d is not None else end
        if e_d > end:
            break
        days = pd.date_range(e_d, last_day, freq="D")
        for d in days:
            if x_d is not None and d == x_d:
                equity[d] = eq * (1 + trade_return(t["side"], t["entry_price"], t["exit_price"]))
            else:
                equity[d] = eq * (1 + trade_return(t["side"], t["entry_price"], close[d]))
        if x_d is not None and x_d <= end:
            eq = eq * (1 + trade_return(t["side"], t["entry_price"], t["exit_price"]))
    equity.iloc[0] = equity.iloc[0] if pd.notna(equity.iloc[0]) else 1.0
    return equity.ffill() - 1


def unknown_assets(tx: pd.DataFrame) -> list[str]:
    """Actifs achetés/vendus sans ticker Yahoo associé."""
    m = tx["kind"].isin(["buy", "sell"]) & (tx["ticker"] == "")
    return sorted(tx.loc[m, "asset"].unique().tolist())


# ---------------------------------------------------------------------------
# PRIX (yfinance) ET CONVERSION EN EUR
# ---------------------------------------------------------------------------

def guess_currency(ticker: str) -> str:
    t = ticker.upper()
    if t.startswith("^"):
        return {"^FCHI": "EUR", "^GDAXI": "EUR", "^STOXX50E": "EUR"}.get(t, "USD")
    if "-" in t and not t.endswith((".DE", ".PA")):
        return t.split("-")[-1]  # crypto : BTC-EUR, ETH-USD...
    suffix = t.rsplit(".", 1)[-1] if "." in t else ""
    return {
        "DE": "EUR", "F": "EUR", "PA": "EUR", "AS": "EUR", "MI": "EUR", "BR": "EUR",
        "MC": "EUR", "LS": "EUR", "VI": "EUR", "HE": "EUR", "IR": "EUR",
        "L": "GBp", "SW": "CHF", "HK": "HKD", "T": "JPY", "TO": "CAD",
    }.get(suffix, "USD")


def asset_class(ticker: str) -> str:
    """Classe d'actif : ASSET_CLASS en priorité, sinon secteur Yahoo traduit, sinon "Autre"."""
    if ticker in ASSET_CLASS:
        return ASSET_CLASS[ticker]
    try:
        import yfinance as yf
        sector = yf.Ticker(ticker).info.get("sector")
        if sector:
            return SECTOR_FR.get(sector, sector)
    except Exception:
        pass
    return "Autre"


def fetch_prices_eur(tickers: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """Cours de clôture convertis en EUR."""
    return fetch_prices(tickers, start, "EUR")


def fetch_prices(tickers: list[str], start: pd.Timestamp, target: str = "EUR") -> pd.DataFrame:
    """
    Télécharge les cours de clôture et les convertit dans la devise `target` (EUR, USD...).
    Renvoie un DataFrame (index = dates, colonnes = tickers).
    """
    import yfinance as yf

    tickers = sorted(set(t for t in tickers if t))
    if not tickers:
        return pd.DataFrame()
    start = pd.Timestamp(start) - pd.Timedelta(days=10)

    data = yf.download(
        tickers, start=start, auto_adjust=False, progress=False, group_by="column", threads=True
    )
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]

    # Devises
    currencies = {}
    for t in tickers:
        cur = None
        try:
            cur = yf.Ticker(t).fast_info.get("currency")
        except Exception:
            cur = None
        currencies[t] = cur or guess_currency(t)

    # Taux de change vers EUR
    needed = sorted({c for c in currencies.values() if c != target})
    fx = {}
    if needed:
        fx_tickers = {c: (f"GBP{target}=X" if c in ("GBp", "GBX") else f"{c.upper()}{target}=X")
                      for c in needed}
        fx_data = yf.download(
            sorted(set(fx_tickers.values())), start=start, auto_adjust=False, progress=False
        )["Close"]
        if isinstance(fx_data, pd.Series):
            fx_data = fx_data.to_frame(name=list(fx_tickers.values())[0])
        fx_data.index = pd.to_datetime(fx_data.index).tz_localize(None).normalize()
        fx_data = fx_data[~fx_data.index.duplicated(keep="last")]
        for c, ft in fx_tickers.items():
            s = fx_data[ft]
            fx[c] = s / 100 if c in ("GBp", "GBX") else s

    out = pd.DataFrame(index=close.index)
    for t in tickers:
        s = close[t] if t in close.columns else pd.Series(np.nan, index=close.index)
        cur = currencies[t]
        if cur != target:
            rate = fx[cur].reindex(out.index.union(fx[cur].index)).ffill().reindex(out.index)
            s = s * rate
        out[t] = s
    return out.sort_index()


# ---------------------------------------------------------------------------
# CALCULS
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    tx: pd.DataFrame
    daily: pd.DataFrame           # value, invested_value, cash, net_deposits, flow, pnl, twr
    positions: pd.DataFrame       # positions ouvertes
    closed: pd.DataFrame          # positions soldées
    summary: dict = field(default_factory=dict)


def _daily_grid(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")


def compute_strategy(name: str, tx: pd.DataFrame, prices: pd.DataFrame,
                     end: pd.Timestamp | None = None) -> StrategyResult:
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    grid = _daily_grid(tx["date"].min(), end)
    tickers = sorted(t for t in tx["ticker"].unique() if t)

    px = prices.reindex(prices.index.union(grid)).sort_index().ffill().bfill().reindex(grid)

    # --- Quantités détenues par jour
    qty = pd.DataFrame(0.0, index=grid, columns=tickers)
    for _, r in tx[tx["kind"].isin(["buy", "sell"])].iterrows():
        sign = 1 if r["kind"] == "buy" else -1
        qty.loc[r["date"]:, r["ticker"]] += sign * r["qty"]
    qty = qty.where(qty.abs() > 1e-9, 0.0)

    # --- Cash et flux externes par jour
    cash_delta = pd.Series(0.0, index=grid)
    flow = pd.Series(0.0, index=grid)  # dépôts (+) / retraits (-)
    for _, r in tx.iterrows():
        d, a = r["date"], r["amount"]
        if r["kind"] == "deposit":
            cash_delta[d] += a; flow[d] += a
        elif r["kind"] == "withdraw":
            cash_delta[d] -= a; flow[d] -= a
        elif r["kind"] == "buy":
            cash_delta[d] -= a
        elif r["kind"] in ("sell", "dividend"):
            cash_delta[d] += a
    # Stratégie sans lignes de dépôt : chaque achat est considéré comme financé
    # par un apport du même montant (et chaque vente comme un retrait).
    if not (tx["kind"] == "deposit").any():
        for _, r in tx[tx["kind"].isin(["buy", "sell"])].iterrows():
            a = r["amount"] if r["kind"] == "buy" else -r["amount"]
            cash_delta[r["date"]] += a
            flow[r["date"]] += a
    cash = cash_delta.cumsum()
    net_deposits = flow.cumsum()

    invested_value = (qty * px[tickers]).sum(axis=1) if tickers else pd.Series(0.0, index=grid)
    value = invested_value + cash

    # --- Performance pondérée par le temps (neutralise les dépôts)
    prev = value.shift(1)
    daily_ret = ((value - flow) / prev - 1).where(prev > 0, 0.0).fillna(0.0)
    twr = (1 + daily_ret).cumprod() - 1

    daily = pd.DataFrame(
        {
            "value": value,
            "invested_value": invested_value,
            "cash": cash,
            "net_deposits": net_deposits,
            "flow": flow,
            "pnl": value - net_deposits,
            "daily_ret": daily_ret,
            "twr": twr,
        }
    )

    # --- Positions (méthode du prix moyen pondéré)
    book = {}
    for _, r in tx[tx["kind"].isin(["buy", "sell"])].iterrows():
        b = book.setdefault(r["ticker"], {"asset": r["asset"], "qty": 0.0, "cost": 0.0,
                                          "realized": 0.0, "bought": 0.0, "sold": 0.0})
        if r["kind"] == "buy":
            b["qty"] += r["qty"]; b["cost"] += r["amount"]; b["bought"] += r["amount"]
        else:
            avg = b["cost"] / b["qty"] if b["qty"] > 0 else 0.0
            sold_cost = avg * r["qty"]
            b["realized"] += r["amount"] - sold_cost
            b["qty"] -= r["qty"]; b["cost"] -= sold_cost; b["sold"] += r["amount"]
            if abs(b["qty"]) < 1e-9:
                b["qty"] = 0.0; b["cost"] = 0.0

    last_px = px.iloc[-1] if len(px) else pd.Series(dtype=float)
    open_rows, closed_rows = [], []
    for t, b in book.items():
        if b["qty"] > 0:
            price = float(last_px.get(t, np.nan))
            val = b["qty"] * price
            open_rows.append(
                {
                    "Actif": b["asset"],
                    "Ticker": t,
                    "Classe": ASSET_CLASS.get(t, "Autre"),
                    "Quantité": b["qty"],
                    "PRU (€)": b["cost"] / b["qty"],
                    "Cours (€)": price,
                    "Coût (€)": b["cost"],
                    "Valeur (€)": val,
                    "P&L latent (€)": val - b["cost"],
                    "P&L latent (%)": (val / b["cost"] - 1) if b["cost"] else np.nan,
                    "P&L réalisé (€)": b["realized"],
                }
            )
        else:
            closed_rows.append(
                {
                    "Actif": b["asset"],
                    "Ticker": t,
                    "Total acheté (€)": b["bought"],
                    "Total vendu (€)": b["sold"],
                    "P&L réalisé (€)": b["realized"],
                }
            )

    positions = pd.DataFrame(open_rows)
    if not positions.empty:
        total_val = positions["Valeur (€)"].sum() + cash.iloc[-1]
        positions["Poids"] = positions["Valeur (€)"] / total_val if total_val else np.nan
        positions = positions.sort_values("Valeur (€)", ascending=False).reset_index(drop=True)
    closed = pd.DataFrame(closed_rows)

    dividends = tx.loc[tx["kind"] == "dividend", "amount"].sum()
    summary = {
        "value": float(value.iloc[-1]),
        "net_deposits": float(net_deposits.iloc[-1]),
        "pnl": float(value.iloc[-1] - net_deposits.iloc[-1]),
        "pnl_pct": float(value.iloc[-1] / net_deposits.iloc[-1] - 1) if net_deposits.iloc[-1] else np.nan,
        "twr": float(twr.iloc[-1]),
        "twr_annual": _annualize(float(twr.iloc[-1]), grid[0], grid[-1]),
        "xirr": xirr(_cashflows_for_xirr(flow, float(value.iloc[-1]), grid[-1])),
        "cash": float(cash.iloc[-1]),
        "realized": float(sum(b["realized"] for b in book.values())),
        "unrealized": float(positions["P&L latent (€)"].sum()) if not positions.empty else 0.0,
        "dividends": float(dividends),
        "max_drawdown": max_drawdown(twr),
        "volatility": float(daily_ret[daily_ret != 0].std() * np.sqrt(252)) if (daily_ret != 0).sum() > 2 else np.nan,
        "start": grid[0],
        "last_price_date": prices.dropna(how="all").index.max() if not prices.empty else None,
    }
    return StrategyResult(name, tx, daily, positions, closed, summary)


def combine_results(results: list[StrategyResult], prices: pd.DataFrame,
                    name: str = "Consolidé") -> StrategyResult:
    """Vue consolidée : on fusionne les transactions de toutes les stratégies."""
    tx = pd.concat([r.tx for r in results], ignore_index=True)
    order = {"deposit": 0, "dividend": 1, "sell": 2, "buy": 3, "withdraw": 4}
    tx["_o"] = tx["kind"].map(order)
    tx = tx.sort_values(["date", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    return compute_strategy(name, tx, prices)


def benchmark_curve(flow: pd.Series, bench_px: pd.Series) -> pd.DataFrame:
    """
    Simule le même calendrier de dépôts/retraits investi à 100 % dans le benchmark.
    Renvoie la valeur simulée et la performance du benchmark sur la période.
    """
    grid = flow.index
    px = bench_px.reindex(bench_px.index.union(grid)).sort_index().ffill().bfill().reindex(grid)
    units = (flow / px).cumsum()
    value = units * px
    perf = px / px.iloc[0] - 1
    return pd.DataFrame({"bench_value": value, "bench_perf": perf})


# ---------------------------------------------------------------------------
# INDICATEURS
# ---------------------------------------------------------------------------

def max_drawdown(twr: pd.Series) -> float:
    idx = 1 + twr
    dd = idx / idx.cummax() - 1
    return float(dd.min()) if len(dd) else np.nan


def _annualize(total: float, start, end) -> float:
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    if years < 1 or total <= -1:
        return np.nan  # pas d'annualisation sur moins d'un an
    return (1 + total) ** (1 / years) - 1


def _cashflows_for_xirr(flow: pd.Series, final_value: float, end) -> list[tuple]:
    cfs = [(d, -a) for d, a in flow[flow != 0].items()]
    cfs.append((pd.Timestamp(end), final_value))
    return cfs


def xirr(cashflows: list[tuple]) -> float:
    """Taux de rendement interne annualisé (rendement réel de ton argent)."""
    if len(cashflows) < 2:
        return np.nan
    t0 = cashflows[0][0]
    times = np.array([(d - t0).days / 365.25 for d, _ in cashflows])
    amts = np.array([a for _, a in cashflows])
    if times[-1] < 1 / 12:
        return np.nan

    def npv(r):
        return np.sum(amts / (1 + r) ** times)

    lo, hi = -0.99, 10.0
    if npv(lo) * npv(hi) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
