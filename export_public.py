"""
Transforme tes Excel privés en fichiers publics sans aucun montant.

    python export_public.py

- Lit tous les .xlsx du dossier prive/   (ce dossier n'est JAMAIS publié, voir .gitignore)
- Écrit un .json par stratégie dans data/ (c'est tout ce que le site lit)
- Les stratégies de trading (format="trades" dans STRATEGY_SETTINGS) publient leur
  journal de trades : dates, niveaux, sens. Aucun montant.

Le JSON public contient uniquement :
  - les ordres : date, actif, sens, prix d'exécution unitaire (ni quantité, ni montant)
  - les poids actuels de chaque ligne en % + le cours de référence du jour de l'export
    (le site fait ensuite évoluer ces poids avec les cours)
  - la courbe de performance de la stratégie en % (aucun montant)

Relance ce script après chaque nouvel ordre, puis publie le dossier data/.
"""

import json
from pathlib import Path

import pandas as pd

import portfolio as pf

ROOT = Path(__file__).parent
PRIVATE_DIR = ROOT / "prive"
PUBLIC_DIR = ROOT / "data"


def check_prices(tx: pd.DataFrame, prices: pd.DataFrame) -> bool:
    """
    Compare le dernier prix d'achat de chaque actif au cours Yahoo du même jour.
    Un écart important = ticker probablement faux (autre produit, autre devise...).
    """
    ok = True
    print("\nContrôle des tickers (prix payé vs cours Yahoo le jour de l'ordre) :")
    orders = tx[tx["kind"].isin(["buy", "sell"])]
    for ticker, g in orders.groupby("ticker"):
        if ticker not in prices.columns or prices[ticker].isna().all():
            print(f"  ❌ {ticker:<9} aucun cours trouvé sur Yahoo -> ticker à changer")
            ok = False
            continue
        r = g.iloc[-1]
        paid = r["amount"] / r["qty"]
        px = prices[ticker].dropna()
        yahoo = float(px[px.index <= r["date"]].iloc[-1]) if (px.index <= r["date"]).any() else float(px.iloc[0])
        gap = paid / yahoo - 1
        flag = "✅" if abs(gap) < 0.05 else "⚠️ "
        if abs(gap) >= 0.05:
            ok = False
        print(f"  {flag} {ticker:<9} {r['date']:%d/%m/%Y}  payé {paid:9.2f} €  |  Yahoo {yahoo:9.2f} €  |  écart {gap:+.1%}")
    return ok


def build_public(name: str, tx: pd.DataFrame) -> dict | None:
    tickers = sorted(t for t in tx["ticker"].unique() if t)
    prices = pf.fetch_prices_eur(tickers, tx["date"].min())
    # On ne garde que des clôtures définitives : le cours du jour (séance en cours ou
    # pas encore consolidé) est exclu. Le site prolonge ensuite avec les cours en direct.
    today = pd.Timestamp.today().normalize()
    if (prices.index >= today).any() and (prices.index < today).any() \
            and not (tx["date"] >= today).any():
        prices = prices[prices.index < today]
    if not check_prices(tx, prices):
        print("\n⛔ Export annulé : corrige les tickers signalés dans TICKER_MAP (portfolio.py).\n")
        return None
    res = pf.compute_strategy(name, tx, prices)

    classes = {t: pf.asset_class(t) for t in tickers}
    pos = res.positions
    cash = max(res.summary["cash"], 0.0)
    total = (pos["Valeur (€)"].sum() if not pos.empty else 0.0) + cash
    last_px = prices.ffill().iloc[-1]

    positions = [
        {
            "ticker": r["Ticker"],
            "name": pf.DISPLAY_NAME.get(r["Ticker"], r["Actif"]),
            "class": classes.get(r["Ticker"], "Autre"),
            "weight": round(r["Valeur (€)"] / total, 4),
            "ref_price": round(float(last_px[r["Ticker"]]), 4),
        }
        for _, r in pos.iterrows()
    ]

    orders = [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "ticker": r["ticker"],
            "name": pf.DISPLAY_NAME.get(r["ticker"], r["asset"]),
            "class": classes.get(r["ticker"], "Autre"),
            "side": r["kind"],
            "price": round(r["amount"] / r["qty"], 4),
        }
        for _, r in tx[tx["kind"].isin(["buy", "sell"])].iterrows()
    ]

    # Courbe de performance en % (pondérée par le temps : les dépôts n'y apparaissent pas)
    as_of = prices.dropna(how="all").index.max()
    twr = res.daily["twr"]
    twr = twr[(twr.index <= as_of) & (twr.index.dayofweek < 5)]

    # Performance de chaque classe d'actifs, calculée comme celle de la stratégie
    # mais uniquement avec les ordres de cette classe (toujours en %, aucun montant).
    class_perf = {}
    for cls in sorted(set(classes.values())):
        tk = [t for t, c in classes.items() if c == cls]
        sub = tx[tx["ticker"].isin(tk)]
        if sub.empty:
            continue
        c_twr = pf.compute_strategy(name, sub, prices[tk]).daily["twr"]
        c_twr = c_twr[(c_twr.index <= as_of) & (c_twr.index.dayofweek < 5)]
        if c_twr.empty:
            continue
        class_perf[cls] = {"dates": [d.strftime("%Y-%m-%d") for d in c_twr.index],
                           "twr": [round(float(v), 6) for v in c_twr.values]}

    return {
        "name": name,
        "start": tx["date"].min().strftime("%Y-%m-%d"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "cash_weight": round(cash / total, 4) if total else 0.0,
        "positions": positions,
        "orders": orders,
        "perf": {
            "dates": [d.strftime("%Y-%m-%d") for d in twr.index],
            "twr": [round(float(v), 6) for v in twr.values],
        },
        "class_perf": class_perf,
    }


def build_trades_public(name: str, settings: dict, trades: pd.DataFrame) -> dict | None:
    """Stratégie de trading : on publie le journal (dates, niveaux, sens), sans aucun montant."""
    ticker = settings["ticker"]
    ccy = settings.get("currency", "EUR")
    prices = pf.fetch_prices([ticker], trades["entry_date"].min(), ccy)
    px = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
    if px.empty:
        print(f"  ❌ {ticker} : aucun cours trouvé sur Yahoo -> ticker à changer dans STRATEGY_SETTINGS")
        return None

    print(f"\nContrôle des niveaux ({ticker}, cours Yahoo de clôture le même jour) :")
    ok = True
    for _, t in trades.iterrows():
        for label, d, lvl in (("entrée", t["entry_date"], t["entry_price"]),
                              ("sortie", t["exit_date"], t["exit_price"])):
            if d is None or pd.isna(d):
                continue
            ref = float(px[px.index <= d].iloc[-1]) if (px.index <= d).any() else float(px.iloc[0])
            gap = lvl / ref - 1
            flag = "✅" if abs(gap) < 0.10 else "⚠️ "
            ok &= abs(gap) < 0.10
            print(f"  {flag} {label:<6} {d:%d/%m/%Y}  niveau {lvl:10.2f}  |  Yahoo {ref:10.2f}  |  écart {gap:+.1%}")
    if not ok:
        print("\n⛔ Export annulé : un niveau est très éloigné du cours Yahoo (date ou niveau à vérifier).\n")
        return None

    def d2s(d):
        return None if d is None or pd.isna(d) else pd.Timestamp(d).strftime("%Y-%m-%d")

    return {
        "name": name,
        "type": "trades",
        "currency": ccy,
        "start": d2s(trades["entry_date"].min()),
        "asset": {"ticker": ticker, "name": settings.get("name", ticker),
                  "class": settings.get("class", "Autre")},
        "trades": [
            {
                "entry_date": d2s(t["entry_date"]),
                "entry_price": round(t["entry_price"], 6),
                "side": t["side"],
                "exit_date": d2s(t["exit_date"]),
                "exit_price": None if pd.isna(t["exit_price"]) else round(t["exit_price"], 6),
            }
            for _, t in trades.iterrows()
        ],
    }


def main():
    PUBLIC_DIR.mkdir(exist_ok=True)
    files = [f for f in sorted(PRIVATE_DIR.glob("*.xlsx")) if not f.name.startswith("~$")]
    if not files:
        print(f"Aucun fichier .xlsx trouvé dans {PRIVATE_DIR}")
        return

    for f in files:
        settings = pf.STRATEGY_SETTINGS.get(f.stem, {})
        out = PUBLIC_DIR / f"{f.stem}.json"
        print(f"\n=== {f.stem} ===")

        if settings.get("format") == "trades":
            trades = pf.load_trades(f)
            data = build_trades_public(f.stem, settings, trades)
            if data is None:
                continue
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            n_open = sum(1 for t in data["trades"] if t["exit_date"] is None)
            print(f"✅ {f.name} -> {out.relative_to(ROOT)} | {len(data['trades'])} trades "
                  f"({n_open} ouvert)")
            continue

        tx = pf.load_transactions(f, settings)
        missing = pf.unknown_assets(tx)
        if missing:
            print(f"⚠️  {f.name} : ticker manquant pour {missing} "
                  "(remplis la colonne Ticker ou TICKER_MAP dans portfolio.py)")
            continue
        data = build_public(f.stem, tx)
        if data is None:
            continue
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        weights = ", ".join(f"{p['name']} {p['weight']:.0%}" for p in data["positions"])
        print(f"✅ {f.name} -> {out.relative_to(ROOT)} | {len(data['orders'])} ordres | {weights}")


if __name__ == "__main__":
    main()
