# Postmortem : resultat_blanc=2.0 n'est PAS un forfait

> **Date** : 2026-03-25
> **Sévérité** : CRITIQUE — 62,197 parties réelles exclues + 295K forfeits inclus
> **Découvert par** : Analyse croisée type_resultat × resultat_blanc × type_competition
> **Statut** : Diagnostic complet, fix en attente

## Résumé

Deux bugs de données superposés :

1. `resultat_blanc = 2.0` traité comme "forfait" et exclu de TOUT (spec V8, 2026-03-21).
   En réalité : **62,197 victoires réelles** en compétitions jeunes FFE.
2. Les vrais forfeits (`type_resultat = forfait_blanc/noir/double_forfait/non_joue`, ~295K lignes)
   ont `resultat_blanc = 0.0 ou 1.0` et sont **inclus dans le training** comme résultats réels.

## Règlement FFE — Barème des points de partie (source de vérité)

| Compétition | Source | Victoire non-U10 | Victoire U10 | Nulle | Défaite |
|-------------|--------|-------------------|--------------|-------|---------|
| A02 Adulte (toutes divisions) | A02 §4.1 | 1 pt | — | X (hors score) | 0 |
| J02 Top/N1/N2 Jeunes éch. 1-6 | J02 §4.1 | **2 pts** | — | X (hors score) | 0 |
| J02 Top/N1/N2 Jeunes éch. 7-8 | J02 §4.1 | — | 1 pt | X (hors score) | 0 |
| J02 N3 Jeunes éch. 1-3 | J02 §4.1 | **2 pts** | — | X (hors score) | 0 |
| J02 N3 Jeunes éch. 4 | J02 §4.1 | — | 1 pt | X (hors score) | 0 |
| ARA N3 Jeunes éch. 1-3 | ARA N3J §4.1 | **2 pts** | — | X (hors score) | 0 |
| ARA N3 Jeunes éch. 4 | ARA N3J §4.1 | — | 1 pt | X (hors score) | 0 |
| ARA Ligue adulte | NIV §4.1 | 1 pt | — | X (hors score) | 0 |

Sources vérifiées : `C:\Dev\chess-app\docs\Règlements ffe\` (A02, J02, R01 2025-26) +
`C:\Dev\ffe_scrapper\data\reglements\Ligues\ARA\`, `BRE\35_Ille-et-Vilaine\`.

**Conclusion** : `resultat_blanc` dans le dataset = points de partie FFE (pas résultat W/D/L).
`2.0` = victoire à 2 pts (jeunes non-U10). `1.0` = victoire à 1 pt (adulte ou jeunes U10).
Les deux = même résultat : le joueur a **gagné** sa partie.

## Crosstab type_resultat × resultat_blanc (1.75M lignes)

| type_resultat | 0.0 | 0.5 | 1.0 | 2.0 |
|--------------|-----|-----|-----|-----|
| victoire_blanc | 0 | 0 | 589,555 | **62,279** |
| victoire_noir | 593,599 | 0 | 0 | 0 |
| nulle | 0 | 193,642 | 0 | 0 |
| forfait_blanc | 43,471 | 0 | 0 | 0 |
| forfait_noir | 0 | 0 | 42,044 | 0 |
| non_joue | 208,981 | 0 | 0 | 0 |
| double_forfait | 2,989 | 0 | 0 | 0 |

`resultat_blanc=2.0` = TOUJOURS `type_resultat=victoire_blanc`. Zéro forfait.
Les vrais forfeits sont `forfait_blanc` (43K, codé 0.0) et `forfait_noir` (42K, codé 1.0).

## Données temporelles

Stable 2002-2026, ~19% des parties jeunes chaque saison. Pas de changement de règlement.
Seul creux 2021 (1.4%) = COVID. Le barème 2pts/victoire jeunes non-U10 est constant.

## Impact sur le training

| Problème | Lignes affectées | Effet |
|----------|-----------------|-------|
| 62K victoires jeunes exclues (2.0=forfait) | 62,197 | Modèle ne voit pas les patterns jeunes |
| 43K forfait_blanc inclus comme défaites (0.0) | 43,471 | Target = loss sans partie jouée |
| 42K forfait_noir inclus comme victoires (1.0) | 42,044 | Target = win sans partie jouée |
| 209K non_joue inclus comme défaites (0.0) | 208,981 | Target = loss sans partie jouée |
| 3K double_forfait inclus (0.0) | 2,989 | Aucune partie jouée |
| Features (win_rate, draw_rate) calculées avec forfeits | Toutes | Rates faussées |

## Root cause

1. Le parser HTML extrait les **points de partie FFE** dans `resultat_blanc`, pas le résultat W/D/L
2. La colonne `type_resultat` identifie explicitement les forfeits — elle n'a pas été utilisée
3. La spec V8 a assumé `2.0 = forfait` sans croiser `type_resultat × resultat_blanc`
4. Le barème FFE jeunes (2pts/victoire) n'était pas documenté dans le projet

## Impact

1. **62K parties exclues** — le modèle ne voit pas les patterns jeunes
2. **Features biaisées** — toutes les rates (win_rate, draw_rate, etc.)
   calculées sans les 62K parties jeunes
3. **Draw rate sous-estimé** — 0.5 existe dans jeunes (8.3% national, 4.1% régional)
   mais 25% de "victoires" sont exclues, faussant les proportions

## Données complètes (crosstab type_resultat × resultat_blanc)

| type_resultat | 0.0 | 0.5 | 1.0 | 2.0 |
|--------------|-----|-----|-----|-----|
| victoire_blanc | 0 | 0 | 589,555 | **62,279** |
| victoire_noir | 593,599 | 0 | 0 | 0 |
| nulle | 0 | 193,642 | 0 | 0 |
| forfait_blanc | 43,471 | 0 | 0 | 0 |
| forfait_noir | 0 | 0 | 42,044 | 0 |
| non_joue | 208,981 | 0 | 0 | 0 |
| double_forfait | 2,989 | 0 | 0 | 0 |

**resultat_blanc=2.0 est TOUJOURS type_resultat=victoire_blanc.** Zéro forfait.
**Les forfeits ont resultat_blanc=0.0 ou 1.0** (pas 2.0).

Stable sur 2002-2026 (~19% des parties jeunes chaque saison, pas de changement de règlement).
1,114 cas en compétition "autre" (Barrages 2002) = aussi des vraies parties.

## Fix

Le ML prédit le résultat de la partie (W/D/L), pas les points de partie FFE.
`resultat_blanc = 2.0` = le joueur a gagné → target = win (2).

Deux corrections nécessaires :

### 1. Filtre des non-parties (FE + training)

```python
# AVANT (wrong) — filtre sur resultat_blanc
df = df[df['resultat_blanc'] != 2.0]  # exclut 62K victoires, garde 295K forfeits

# APRÈS (correct) — filtre sur type_resultat
# Pour le TRAINING (targets) : exclure les non-parties
NON_PLAYED = {'forfait_blanc', 'forfait_noir', 'double_forfait', 'non_joue'}
df_played = df[~df['type_resultat'].isin(NON_PLAYED)]

# Pour le FE (features) : garder les forfeits pour calculer taux_forfait, taux_presence, etc.
# Puis filtrer pour le training après calcul des features
```

### 2. Recodage resultat_blanc (target mapping)

```python
# AVANT (wrong)
TARGET_MAP = {0.0: 0, 0.5: 1, 1.0: 2}  # 2.0 non mappé → exclu

# APRÈS (correct)
TARGET_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2}  # 2.0 = victoire jeunes = win
```

### Fichiers à modifier

| Fichier | Modification |
|---------|-------------|
| `scripts/features/helpers.py` | `FORFAIT_RESULT = 2.0` → supprimer. `filter_played_games` → utiliser `type_resultat` |
| `scripts/features/helpers.py` | `compute_wdl_rates` → compter `2.0` comme win |
| `scripts/kaggle_trainers.py` | `_split_xy` → `TARGET_MAP` inclut `2.0: 2` |
| `scripts/cloud/fe_kaggle.py` | Vérifier que FE conserve les forfeits pour features puis filtre pour splits |
| Tout fichier appelant `exclude_forfeits()` | Remplacer par filtre `type_resultat` |

## Leçons

- `type_resultat` identifie les forfeits EXPLICITEMENT — l'utiliser, pas `resultat_blanc`
- TOUJOURS croiser les colonnes (type_resultat × resultat_blanc) avant de décider d'exclure
- TOUJOURS vérifier par saison si le comportement est stable ou lié à un changement de règlement
- TOUJOURS lire les règlements officiels FFE (A02, J02) avant d'interpréter les données
- Le barème FFE jeunes (2 pts/victoire non-U10) est constant depuis 2002
- Les points de partie ≠ le résultat de la partie. Le ML prédit W/D/L, pas les points FFE
