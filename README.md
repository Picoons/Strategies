# Vitrine de stratégies d'investissement

Site Streamlit qui affiche, pour chaque stratégie :
- la répartition actuelle en % (par classe d'actif et par ligne) et l'écart à la cible ;
- l'historique des ordres : date, sens, prix d'exécution, variation depuis l'ordre ;
- chaque ordre placé sur le graphique de cours de l'actif.

Aucun montant ni aucune quantité n'est publié.

## Structure

```
app.py              le site
portfolio.py        lecture des Excel, tickers Yahoo, calculs
export_public.py    Excel privés -> JSON publics
prive/              TES EXCEL (ignoré par git, jamais publié)
data/               JSON publics lus par le site
requirements.txt
```

## Utilisation

```bash
pip install -r requirements.txt
python export_public.py      # à relancer après chaque nouvel ordre
streamlit run app.py
```

Nouvelle stratégie : dépose l'Excel dans `prive/`, ajoute les tickers des nouveaux
actifs dans `TICKER_MAP` et `ASSET_CLASS` (portfolio.py), la cible et la description
dans `TARGET_ALLOCATION` / `DESCRIPTIONS` (app.py), puis relance l'export.

## Mise en ligne (Streamlit Community Cloud)

Pousse le dossier sur GitHub (le .gitignore exclut `prive/` et les `.xlsx`),
puis crée l'app sur share.streamlit.io en pointant sur `app.py`.
Vérifie avec `git status` qu'aucun Excel n'est suivi avant le premier push.
