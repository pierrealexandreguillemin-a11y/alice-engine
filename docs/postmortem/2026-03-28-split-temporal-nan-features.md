# Postmortem: 61 features 100% NaN sur valid/test

> **Date**: 28 Mars 2026
> **Severite**: CRITIQUE — cause racine des 116 features "mortes" depuis v1
> **Impact**: es_mae quality gate bloque depuis v9, features equipe/standings invisibles

## Symptome

61/220 features sont 100% NaN sur valid et test mais OK sur train.
Inclut: position, zone_enjeu, noyau, profondeur, rotation, promu/relegue,
win_rate_home, club_level, reinforcement_rate, marge_100, standings complets.

## Cause racine

`feature_engineering.py` lignes 219 et 230 :

```python
# Valid: history EXCLUT la saison 2023
valid_history = df[df["saison"] < valid_raw["saison"].min()]  # < 2023 = ≤ 2022

# Test: history EXCLUT les saisons 2024+
test_history = df[df["saison"] <= test_raw["saison"].min() - 1]  # ≤ 2023
```

Les features equipe (standings, club_behavior, noyau, club_level) sont indexees
par `(equipe, saison)`. Le merge echoue quand la saison du split n'est pas dans
l'historique → NaN silencieux.

Pendant ce temps, le train fait :
```python
train_history = df[df["saison"] <= train_raw["saison"].max()]  # ≤ 2022
```
→ inclut la saison courante → features disponibles → ca marche.

## Pourquoi pas detecte avant

1. Les arbres (CatBoost/XGBoost/LightGBM) gerent les NaN silencieusement
   (direction par defaut au noeud)
2. Le modele entraine sur train (features OK) et evalue sur test (features NaN)
   → les features equipe ne contribuent jamais a l'eval → permutation importance = 0
3. On a attribue les 116 features "mortes" a d'autres causes (profondeur arbre,
   match-level constant, features individuelles pas differenciees)
4. Aucun check de completeness par split (l'industrie recommande Deequ/Evidently)

## Fix

Aligner valid/test sur le meme comportement que train :

```python
# AVANT (BROKEN):
valid_history = df[df["saison"] < valid_raw["saison"].min()]
test_history = df[df["saison"] <= test_raw["saison"].min() - 1]

# APRES (FIXED):
valid_history = df[df["saison"] <= valid_raw["saison"].max()]
test_history = df[df["saison"] <= test_raw["saison"].max()]
```

## Caveat : leakage intra-saison

Avec ce fix, une partie ronde 3 de la saison 2023 a un historique qui inclut
les rondes 4+ de 2023. C'est le MEME comportement que le train (une partie
ronde 3 de 2020 a un historique qui inclut les rondes 4+ de 2020).

Ce leakage intra-saison est ACCEPTE car :
- Les standings sont merges par (equipe, saison, ronde) → pas de leakage ronde
- Les features joueur utilisent des rolling windows passees
- Les features club sont des aggregats saison (leakage mineur, acceptable)
- La correction per-ronde necessiterait 1.4M calculs individuels (trop couteux)

## Lecon

**TOUJOURS verifier le taux de NaN PAR SPLIT avant d'entrainer.**
Un check de 5 lignes aurait evite 2 semaines de debug.
