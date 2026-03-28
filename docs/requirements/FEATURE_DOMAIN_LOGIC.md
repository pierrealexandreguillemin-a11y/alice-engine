# Logique Metier des Features ML - ALICE Engine

<!--
LLM CONTEXT: Ce document est le companion analytique de FEATURE_SPECIFICATION.md.
Il explique POURQUOI chaque feature predit les resultats de boards d'echecs interclubs,
comment les features s'alignent avec la litterature ML sport, et quels differentiels
(blanc-noir, dom-ext) sont necessaires. A lire AVANT tout travail de feature engineering.

Audience: LLM assistant + developpeur ML. Utiliser comme reference pour:
- Justifier l'ajout/suppression de features
- Comprendre la logique capitaine FFE
- Comparer avec la litterature multisports
- Identifier les features manquantes
-->

> **Document Type**: Domain Analysis (ISO 15289)
> **Version**: 1.0.0
> **Date**: 27 Mars 2026
> **Companion de**: `FEATURE_SPECIFICATION.md` (spec formelle: types, plages, ISO 5259)
> **Resultats training**: v15-v16 sur `feat/v8-multiclass-features`

---

## Sommaire

1. [Principe fondamental: ALICE vs litterature multisports](#1-principe-fondamental)
2. [Taxonomie: Prediction board vs Composition CE](#2-taxonomie)
3. [Features VIVANTES (17 validated v16)](#3-features-vivantes)
4. [Features MORTES: diagnostic metier par categorie](#4-features-mortes)
5. [Features DIFFERENTIELLES manquantes](#5-differentiels-manquants)
6. [Features INTERACTION board x match](#6-interactions)
7. [Features A CREER (non existantes)](#7-features-a-creer)
8. [References et sources](#8-references)

---

## 1. Principe fondamental

### 1.1 Le probleme ALICE

ALICE predit P(victoire), P(nulle), P(defaite) pour un **board specifique** :
joueur blanc (Elo, forme, historique) vs joueur noir, dans un contexte de match
(equipe dom vs ext, classement, enjeu, composition).

Le CE (Composition Engine) utilise ces probabilites pour optimiser l'allocation
joueurs -> echiquiers sous contraintes FFE.

### 1.2 Ce que dit la litterature multisports

Les modeles qui gagnent les challenges de prediction sport
(Hubacek et al. 2019, RPS=0.1925 ; arXiv:2309.14807, CatBoost champion)
partagent **3 principes** :

**P1 - Features RELATIVES** : la force d'une equipe n'a de sens que RELATIVE
a l'adversaire. `diff_form = form_A - form_B` > `form_A` + `form_B` separes.
Un arbre depth=4 gaspille 2 splits a soustraire deux features individuelles.
Avec le differentiel, 1 split suffit, les 3 restants explorent les interactions.

**P2 - Niveau du matchup** : les features doivent decrire le MATCHUP, pas l'entite.
"equipe A est 3eme" est faible. "equipe A est 5 places devant B" est un signal.

**P3 - Le feature engineering > le choix du modele** : "90% des papiers listent
des features plus riches comme premier objectif futur" (review arXiv:1912.11762).

### 1.3 Le gap ALICE (v16, mars 2026)

196 features dont **5 relatives** (diff_elo, diff_titre, avg_elo, elo_proximity, draw_rate_prior).
116 features mortes en training (0 permutation importance, 0 SHAP cross-modele).
17 features validated — toutes sont soit relatives (diff_elo) soit des taux de performance
joueur (expected_score_recent, draw_rate) qui agissent comme des proxys relatifs via le
residual learning (la baseline Elo capture deja le niveau absolu).

**Diagnostic** : les features mortes sont individuelles (_blanc, _noir, _dom, _ext)
et match-level (constantes pour les 8 boards). Il manque les differentiels et les
interactions board x match.

---

## 2. Taxonomie

### 2.1 Prediction board (modele ML)

Features utiles pour predire le resultat d'un board specifique.
Doivent varier PAR BOARD ou capturer un differentiel de matchup.

### 2.2 Composition CE (solveur OR-Tools V9)

Features utiles pour le CE : contraintes FFE, eligibilite joueur.
PAS pour la prediction — elles regissent QUI peut jouer OU.

| Feature | Prediction ML | Composition CE | Raison |
|---------|:---:|:---:|---|
| diff_elo | X | | Differentiel de force |
| expected_score_recent | X | | Forme joueur |
| zone_enjeu (diff) | X | | Pression contextuelle |
| est_dans_noyau | | X | Contrainte selection (A02 3.7.f) |
| matchs_avant_brulage | | X | Contrainte brulage (A02 3.5) |
| nb_mutes_restants | | X | Quota transferts (A02 3.7.g) |
| joueur_brule | | X | Eligibilite binaire |
| pct_noyau_equipe | X (faible) | X | Stabilite equipe + contrainte |

### 2.3 Noyau: composition, PAS prediction

**Definition FFE (A02 Art. 3.7.f)** : un joueur est dans le noyau s'il a deja
joue au moins 1 fois pour cette equipe cette saison. Minimum 50% noyau en national.

`est_dans_noyau` = eligibilite, pas qualite. Un joueur noyau n'est pas meilleur.
Le CE l'utilise comme contrainte de composition. Le ML ne doit PAS l'utiliser
comme predicteur (ce serait apprendre un artefact administratif).

En revanche, `pct_noyau_equipe` (% de l'equipe qui est noyau) mesure la STABILITE
d'equipe — une equipe a 80% noyau a plus de cohesion qu'une a 50%.
Ce % est un signal match-level faible mais reel.

---

## 3. Features VIVANTES (17 validated v16)

Les 17 features validated par permutation cross-modele (CatBoost + XGBoost + LightGBM)
dans v16 (alpha=0.7, donnees propres). Toutes font sens metier.

### 3.1 Taux de performance joueur (COEUR du signal)

| Feature | Logique metier | Pourquoi ca marche |
|---------|---------------|-------------------|
| `expected_score_recent_blanc/noir` | Forme sur 5 derniers matchs (meme niveau competition) | Proxy le plus direct de "comment joue ce joueur EN CE MOMENT" |
| `win_rate_recent_blanc` | % victoires recentes | Confiance, momentum |
| `draw_rate_recent_blanc/noir` | Tendance a la nulle | Certains joueurs "jouent pour la nulle" |
| `draw_rate_blanc/noir` | Taux de nulles career (3 saisons, stratifie) | Profil joueur fondamental |

**Pourquoi elles vivent** : elles corrigent l'Elo. Un joueur 1800 Elo en forme
(expected_score=0.75) est different du meme joueur en crise (expected_score=0.40).
L'Elo baseline ne capte pas les fluctuations de forme.

### 3.2 Differentiels Elo

| Feature | Logique metier | Pourquoi ca marche |
|---------|---------------|-------------------|
| `diff_elo` | Force relative directe | #1 permutation importance. LE differentiel. |
| `elo_proximity` | 1 - \|diff_elo\|/800 | Probabilite de nulle (plus les Elos sont proches, plus de nulles) |

### 3.3 Draw-specific

| Feature | Logique metier | Pourquoi ca marche |
|---------|---------------|-------------------|
| `draw_rate_equipe_ext` | L'equipe ext fait des nulles | Profil defensif d'equipe |
| `h2h_win_rate` | Historique face-a-face | Signal fort mais sparse (0.8% couverture) |
| `h2h_draw_rate` | Nulles entre ces deux joueurs | Idem |

### 3.4 Position et contexte

| Feature | Logique metier | Pourquoi ca marche |
|---------|---------------|-------------------|
| `decalage_position_blanc/noir` | Joueur place plus haut/bas que son echiquier habituel | Le capitaine a CHOISI de le deplacer — signal strategique |
| `draw_trend_noir` | Tendance recente aux nulles de noir | Momentum de draw |
| `ronde_normalisee` | Progression de la saison | Enjeux croissants |
| `division` | Niveau de competition | National vs regional (draw rates tres differents) |
| `ronde` | Numero de ronde | Temporalite |

---

## 3. Audit features orphelines (2026-03-27)

### 3.1 Code mort : modules CE jamais appeles (~600 lignes)

| Module | Fonction | Lignes | Statut | Feature utile pour ML ? |
|---|---|---|---|---|
| `ce/scenarios.py` | `calculate_scenario_features()` | ~144 | **DEAD CODE** | `urgence_score` [0,1] OUI — remplace `match_important` binaire |
| `ce/urgency.py` | `calculate_urgency_features()` | ~134 | **DEAD CODE** | `montee_possible`, `maintien_assure` OUI — signaux tactiques |
| `ce/transferability.py` | `calculate_transferability()` | ~125 | **DEAD CODE** | `transfer_score` PEUT-ETRE — potentiel mouvement joueur |

**Action** : brancher `urgence_score` dans le FE pipeline (P1). C'est un flottant continu
bien meilleur que `match_important` (81% = 1, inutile).

### 3.2 Features categoriques droppees silencieusement

`kaggle_trainers.py:65` : `select_dtypes(include=["int64","float64"...])` elimine
les strings non encodees. Features impactees :

| Feature | Type | Dans liste encodage ? | Statut |
|---|---|---|---|
| `couleur_preferee_blanc/noir` | string | **NON** | **DROPPEE** — jamais vue par le modele |
| `pressure_type_blanc/noir` | string | A VERIFIER | Potentiellement droppee |
| `zone_enjeu_dom/ext` | string | CATBOOST_CAT ✓ | OK si label-encodee |
| `win_trend_blanc/noir` | string | ADVANCED_CAT ✓ | OK |
| `draw_trend_blanc/noir` | string | ADVANCED_CAT ✓ | OK |

**Action** : ajouter `couleur_preferee` a ADVANCED_CAT (P1). Ou mieux : calculer
`color_match` (booleen) qui subsume `couleur_preferee` + board parity + est_domicile.

### 3.3 Features numeriques : inclusion AUTOMATIQUE

Toute feature numerique mergee dans le DataFrame passe dans X_train via `select_dtypes`.
Les listes explicites (CATEGORICAL, BOOL, ADVANCED_CAT) ne controlent que l'ENCODAGE.
Confirme par v16 : `decalage_position_blanc` est VALIDATED alors qu'elle n'est dans aucune liste.

→ Les differentiels numeriques seront automatiquement inclus. Pas besoin de modifier
kaggle_constants.py pour les features numeriques.

### 3.5 Fantomes dans kaggle_constants.py

Features listees dans ADVANCED_CAT_FEATURES mais JAMAIS COMPUTEES par le FE pipeline.
L'encodeur les skip silencieusement (`if col not in train.columns: continue`).
A nettoyer pour eviter la confusion :

- `data_quality_blanc/noir` : jamais compute → retirer
- `elo_type_blanc/noir` : jamais compute → retirer (redondant avec Elo brut)
- `categorie_blanc/noir` : jamais compute → retirer OU implementer (k_coefficient proxy)

Asymetrie : `zone_enjeu_dom` dans CATBOOST_CAT, `zone_enjeu_ext` dans ADVANCED_CAT
→ encodage different pour la meme feature. Corriger en ajoutant ext dans CATBOOST_CAT.

### 3.6 Modules jamais appeles

- `extract_adversaire_niveau()` : adversaire_niveau_dom/ext — utile, a brancher
- `extract_temporal_features()` : phase_saison, ronde_normalisee — verifier si ronde_normalisee
  arrive par un autre chemin
- `ce/scenarios.py` : urgence_score [0,1] — meilleur que match_important (81% positif),
  proxy `urgence_proxy = zone_critique × ronde_normalisee` dans differentials.py
- `ce/urgency.py`, `ce/transferability.py` : CE-only, hors scope ML

### 3.4 Training-serving skew (FTI pattern, Hopsworks)

Le module de differentiels DOIT etre utilisable en batch (FE pipeline) ET en online
(inference.py). "If there is skew between transformations, you will have model
performance bugs very hard to identify" (KDnuggets/Hopsworks 2023).

**Contrainte design** : `differentials.py` prend un DataFrame (1 ou 1M lignes),
ajoute les colonnes, retourne le DataFrame. Pur, sans etat, vectorise.
Pas de dependance au dataset complet.

Sources:
- Hopsworks FTI: https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines
- KDnuggets: https://www.kdnuggets.com/2023/09/hopsworks-unify-batch-ml-systems-feature-training-inference-pipelines

---

## 3bis. Pattern ML → Feature ALICE : mapping complet

Pour chaque groupe de features, on applique le pattern ML de la litterature
AU contexte specifique d'ALICE (echecs interclubs FFE, prediction board-level,
utilisation par le CE).

### FORME / MOMENTUM

**Pattern ML** : NBA (PMC11265715) — "features for home and away teams are
subtracted from each other". Soccer (arXiv:2309.14807) — Form2 = pts last 3
matches / 9, weighted streak. TOUJOURS relatif a l'adversaire.

**ALICE actuel** : `expected_score_recent_blanc/noir` (5 matchs, stratifie par
competition) — individuels, pas differencies. C'est comme si le NBA paper
gardait "home_FG%=45%" et "away_FG%=42%" separes au lieu de "diff_FG%=3%".

**Logique capitaine** : le capitaine aligne un joueur EN FORME (ESR=0.75)
contre un adversaire EN CRISE (ESR=0.40). C'est le DELTA qu'il evalue,
pas les valeurs absolues. "Mon joueur gagne 3 sur 5, le leur perd 4 sur 5
→ mon joueur est favori."

**Utilisation CE** : E[score] = P(win) + 0.5×P(draw). Si le modele capte mieux
le delta de forme → P(win) plus precis → CE optimise mieux l'allocation.

**Fix** :
- P0 : `diff_form = ESR_blanc - ESR_noir` (le differentiel fondamental)
- P1 : `form_home_blanc` = ESR calcule uniquement sur matchs a domicile du joueur
  (separer dom/ext dans le calcul de forme — le joueur performe-t-il pareil partout ?)
- P2 : `form_adjusted = ESR * (mean_opponent_elo / mean_elo_global)` (opponent-adjusted,
  source: arXiv:2410.21484 "strength of opposition" adjustment)

### CLASSEMENT / ZONE D'ENJEU

**Pattern ML** : Soccer (arXiv:2309.14807) — L_up_i (points behind leader),
L_down_i (gap from bottom), DIFFERENCIES entre equipes. Calendar quarter.
Cricket (arXiv:2410.21484) — "strength of opponent adjusted ratings".

**ALICE actuel** : `position_dom/ext`, `ecart_premier/dernier`, `zone_enjeu_dom/ext`.
Tous individuels, tous match-level. `match_important` = 81% positif (inutile).

**Logique capitaine** : le capitaine ne pense PAS "on est 3eme". Il pense
"on est 5 places devant eux" ou "ils se battent pour le maintien, on est
tranquilles" → c'est le DIFFERENTIEL et l'ASYMETRIE de zone qui comptent.

**Utilisation CE** : V9 strategy modes dependent directement de zone_enjeu.
Mode "tactique ronde" = max P(victoire match) si equipe en zone danger.
Mode "agressif" = max E[score] equipe prioritaire si en course_titre.
→ zone_enjeu est une ENTREE STRATEGIQUE du CE, pas juste un contexte.

**Fix** :
- P0 : `diff_position = position_dom - position_ext` (qui est devant ?)
- P1 : `zone_asymmetry` = one-hot de zone_dom × zone_ext (16 combos)
  - danger×montee = affrontement direct, maximum de tension
  - danger×mi_tableau = equipe dom desesperee, ext sereine → pression asymetrique
  - montee×montee = choc au sommet, les deux jouent "a fond"
  - mi_tableau×mi_tableau = faible enjeu → plus de nulles attendues
- P1 : remplacer `match_important` (binaire inutile) par `urgence_score`
  continu (deja calcule dans `ce/scenarios.py`, pas branche en ML)

### COMPOSITION EQUIPE

**Pattern ML** : Soccer — "total transfer value of benched players" (profondeur
banc). NBA — rotation patterns, minutes distribution. Tous DIFFERENCIES.
Review arXiv:1912.11762 — "team stability affects match variance".

**ALICE actuel** : `noyau_stable_dom/ext`, `rotation_effectif_dom/ext`,
`profondeur_effectif_dom/ext`, `club_utilise_marge_100_dom`. Individuels, match-level.

**Logique capitaine** : "mon equipe a 20 joueurs disponibles, la leur en a 8.
On peut tourner, eux non. Sur 7 rondes, on tient mieux." Le capitaine pense
en AVANTAGE DE PROFONDEUR, pas en profondeur absolue.

`club_utilise_marge_100` est un signal de SOPHISTICATION TACTIQUE unique aux
echecs : le club exploite deliberement la marge de 100 pts Elo (A02 3.6.e)
pour placer ses joueurs sur des echiquiers strategiques plutot que par Elo strict.
Un club qui fait ca a un capitaine qui REFLECHIT a la composition. Les autres
alignent par defaut.

**Utilisation CE** : la profondeur est une CONTRAINTE du CE (plus de joueurs =
plus de combinaisons possibles). La stabilite affecte la VARIANCE des predictions
(equipe stable → predictions plus fiables → CE peut optimiser plus agressivement).

**Fix** :
- P0 : `diff_profondeur = profondeur_dom - profondeur_ext` (avantage de banc)
- P0 : `diff_stabilite = noyau_stable_dom - noyau_stable_ext` (cohesion relative)
- P1 : `marge100_actif = club_utilise_marge_100_dom × decalage_position_blanc`
  (club strategique ET ce joueur est place deliberement → signal fort)
- P2 : `diff_rotation × ronde_normalisee` (la rotation en fin de saison = fatigue
  ou gestion ; en debut de saison = experimentation)

### VASES COMMUNIQUANTS / HIERARCHIE CLUB

**Pattern ML** : PAS d'equivalent direct en soccer/basket (transferts marche ≠
mouvements internes club). Le plus proche : soccer "mid-season reinforcement"
(joueur signe en janvier pour renforcer). ALICE a un avantage unique ici.

**ALICE actuel** : `reinforcement_rate_dom/ext` (% rondes avec joueurs d'autres
equipes du club), `elo_moyen_evolution_dom/ext`, `joueur_promu/relegue`,
`player_team_elo_gap`.

**Logique capitaine** : "l'equipe 1 tire de l'equipe 2 depuis 3 rondes → equipe 1
en crise, equipe 2 affaiblie." OU : "un joueur de N1 descend jouer en N3 →
renforcement massif pour N3." Les vases communiquants sont LA specificite
des interclubs multi-equipes. Aucun autre sport n'a cette dynamique.

**Utilisation CE** : le CE V9 multi-equipe GERE directement ces mouvements.
Il doit savoir : si je prends joueur X de l'equipe 2 pour l'equipe 1,
quel est l'IMPACT sur les deux equipes ? → le modele ML doit capturer
l'effet d'un joueur promu/relegue sur sa performance.

**Fix** :
- P1 : `diff_reinforcement = reinforcement_rate_dom - reinforcement_rate_ext`
  (qui est plus desespere ? qui tire le plus de ses reserves ?)
- P1 : `diff_elo_evolution = elo_moyen_evolution_dom - elo_moyen_evolution_ext`
  (une equipe se renforce, l'autre s'affaiblit → momentum d'equipe)
- P2 : `promu_context = joueur_promu × player_team_elo_gap × diff_elo`
  (triple interaction : promu + au-dessus du niveau moyen + face a adversaire fort
  → vrai signal de renforcement delibere vs joueur deplace par defaut)

### COULEUR / PIECES

**Pattern ML** : Specifique aux echecs. Pawnalyze 2022 — LightGBM capte des
patterns non-lineaires que la formule Elo ne voit pas. FIDE reconnu : blancs
ont un avantage statistique (+54% toutes categories confondues).

**ALICE actuel** : `win_rate_white/black`, `win_adv_white`, `draw_adv_white`,
`couleur_preferee` — par joueur, 3 saisons rolling. Sparse (beaucoup de joueurs
n'ont pas assez de matchs par couleur). Pas d'interaction avec le board actuel.

**Logique capitaine** : convention FFE — echiquiers impairs = blancs pour dom.
Le capitaine SAIT quelle couleur chaque board aura. Il place les joueurs
forts aux blancs sur les echiquiers impairs a domicile. C'est un CHOIX
DELIBERE de composition, pas un hasard.

**Utilisation CE** : le CE V9 doit connaitre la couleur de chaque board
pour optimiser. Un joueur +0.15 aux blancs place sur un echiquier pair a
domicile (donc noirs) PERD cet avantage. Le CE peut reordonner pour
maximiser les color_match.

**Fix** :
- P1 : `color_actual = "blanc" if (est_domicile AND echiquier % 2 == 1) else "noir"`
  → couleur EFFECTIVE sur ce board (deterministe, pas une feature ML mais un calcul)
- P1 : `color_match = (couleur_preferee == color_actual)` → booleen : le joueur
  a-t-il sa couleur preferee ? C'est le signal qui compte, pas le profil brut.
- P2 : `color_advantage_effective = win_adv_white × (1 if color_actual=="blanc" else -1)`
  → l'avantage de couleur SIGNE selon la couleur reelle sur ce board

### PRESSION / CLUTCH

**Pattern ML** : NBA — "clutch stats" (dernières 5 min, score serre).
Soccer (arXiv:2309.14807) — weighted streak. Cricket (CAMP model) —
performance par game situation. CLE : la pression est RELATIVE au contexte
ET a l'adversaire.

**ALICE actuel** : `clutch_win/draw`, `pressure_type`, `win/draw_rate_pression`.
Definition : zone_enjeu IN (montee, danger) → 54% des matchs "sous pression"
(trop large, pas discriminant). SHAP > 0 mais permutation NEGATIVE (overfit).

**Logique capitaine** : "je SAIS que mon joueur craque sous pression, je ne
le mets pas echiquier 1 quand on joue le maintien." C'est un signal
PERSONNEL (clutch vs choke) que le capitaine integre INTUITIVEMENT.
Le differentiel clutch vs adversaire est la formalisation de : "mon joueur
monte en pression, le leur descend → avantage maximal dans ce match crucial."

**Utilisation CE** : mode "risk-adjusted" = max E[score] - lambda×Var[score].
Un joueur clutch a MOINS de variance sous pression → le CE le prefere
en mode conservateur. Un joueur choke a PLUS de variance → le CE l'evite.

**Fix** :
- P1 : `diff_clutch = clutch_win_blanc - clutch_win_noir` (differentiel pression)
- P2 : affiner pression : utiliser `urgence_score` de `ce/scenarios.py`
  (continu [0,1]) au lieu du binaire zone_enjeu. Ou :
  `pression_reelle = zone_danger AND ronde >= 7 AND ecart_dernier <= 2`
  (relegation imminente = ~10% des matchs, bien plus discriminant que 54%)
- P2 : `clutch_in_context = clutch_win_blanc × urgence_score_dom`
  (le clutch du joueur PONDERE par l'urgence reelle du match)

### HEAD-TO-HEAD

**Pattern ML** : Hubacek 2019 — H2H inclus dans features soccer. Standard dans
TOUS les sports. Inherement relatif (paire specifique).

**ALICE actuel** : `h2h_win_rate`, `h2h_draw_rate`, `h2h_nb_confrontations`,
`h2h_exists`. Seuil >= 3 confrontations. Couverture 0.8% (sparse).

**Logique capitaine** : "A perd TOUJOURS contre B, je ne les mets pas face a face."
C'est LA feature du capitaine. Quand elle existe, c'est le signal le plus fort.

**Utilisation CE** : le CE V9 genere des scenarios adverses (ALI). Pour chaque
scenario, il evalue chaque paire joueur-adversaire. Le H2H est l'info la plus
directe pour cette paire specifique.

**Fix** :
- Bien construit, pas de differentiel a ajouter (deja relatif)
- P2 : baisser seuil de 3 a 2 confrontations ? (plus de couverture, plus de bruit)
- P2 : `h2h_proxy` quand H2H n'existe pas : utiliser les matchups contre des
  joueurs de niveau similaire (diff_elo dans ±50 de l'adversaire actuel)

### ELO TRAJECTORY / MOMENTUM

**Pattern ML** : Soccer (arXiv:2309.14807) — win_trend, draw_trend categories.
Glicko-2 (Glickman 2001) — rating deviation capture l'incertitude temporelle.
Maia Chess (Microsoft) — players identifiables from 10 games → profil individuel
capturable en ML.

**ALICE actuel** : `elo_trajectory` (progression/regression/stable, seuil 50 pts),
`momentum` (normalise [-1,1]). Par joueur, 6 matchs rolling.

**Logique capitaine** : "ce junior a gagne 100 pts Elo en 3 mois → il est
BEAUCOUP plus fort que son Elo actuel ne dit." L'Elo a un RETARD sur la
forme reelle du joueur. La trajectoire capture ce retard.

**Utilisation CE** : un joueur en progression a un Elo qui SOUS-ESTIME sa force
actuelle → le CE devrait le placer plus haut que son Elo ne le suggere.
Inverse pour un joueur en regression.

**Fix** :
- P1 : `diff_momentum = momentum_blanc - momentum_noir` (qui progresse plus vite ?)
- P2 : `trajectory × k_coefficient` : jeune en progression (K=40 + trajectory_up)
  = Elo TRES sous-estime (le K amplifie le retard). Veteran en regression (K=10
  + trajectory_down) = Elo presque a jour (le K freine la descente).
  → `elo_lag_proxy = momentum × k_coefficient / 20` (normalisé)

### PRESENCE / DISPONIBILITE

**Pattern ML** : NBA — "rest days", "back-to-back games". Soccer (arXiv:2410.21484)
— "player fatigue", "schedule factors". Key : fraicheur RELATIVE entre adversaires.

**ALICE actuel** : `taux_presence_saison/global`, `derniere_presence`, `regularite`,
`rondes_manquees_consecutives`. Individuels, non differencies.

**Logique capitaine** : "mon joueur n'a pas joue depuis 4 rondes, il est rouille.
Le leur joue toutes les semaines." → le capitaine evalue la FRAICHEUR RELATIVE.

**Utilisation CE** : la presence recente affecte la FIABILITE de la prediction.
Un joueur avec 10 matchs recents a des stats fiables. Un joueur avec 2 matchs
→ stats bruitees → CE devrait etre plus conservateur.

**Fix** :
- P1 : `diff_derniere_presence = derniere_presence_blanc - derniere_presence_noir`
  (rouille relative : positif = blanc plus rouille que noir)
- P1 : `diff_taux_presence = taux_presence_blanc - taux_presence_noir`
  (confiance capitaine relative : le capitaine aligne-t-il plus ce joueur que l'adversaire ?)
- P2 : `data_confidence = min(nb_matchs_forme_blanc, nb_matchs_forme_noir)`
  (fiabilite du matchup : si l'un des deux a < 3 matchs, les stats sont bruitees)

---

## 4. Features MORTES: diagnostic metier par categorie

### 4.1 CLASSEMENT (14/14 mortes)

`zone_enjeu_dom/ext`, `position_dom/ext`, `ecart_premier/dernier`, `points_cumules`, `nb_equipes`, `adversaire_niveau`, `match_important`

**Le capitaine y pense ?** OUI. C'est sa premiere info avant de composer.
Un match montee vs maintien est DIFFERENT d'un match mi-tableau.

**Pourquoi mortes ?**
- **Match-level** : identiques pour les 8 boards du meme match
- **Pas differenciees** : `position_dom=3` et `position_ext=8` separement ne disent rien
  a un arbre depth=4. `diff_position=-5` (dom 5 places devant) dit tout en 1 split.
- `match_important` : 81.3% = 1, quasi-constant
- `zone_enjeu` : string catégoriel label-encodes en entiers arbitraires

**Fix** : `diff_position`, `diff_points_cumules`, one-hot zone_enjeu, interaction
asymetrique (une equipe danger, l'autre montee = match desequilibre).

### 4.2 COMPOSITION EQUIPE (16/16 mortes)

`nb_joueurs_utilises`, `rotation_effectif`, `noyau_stable`, `profondeur_effectif`,
`renforce_fin_saison`, `club_utilise_marge_100`, `win_rate_home_dom/ext`, `draw_rate_home_dom/ext`

**Le capitaine y pense ?** OUI. Il sait que son equipe est profonde ou fragile.

**Pourquoi mortes ?**
- **Match-level** : memes valeurs pour les 8 boards
- **Pas differenciees** : `profondeur_dom=15` vs `profondeur_ext=8` separement
  ne disent rien. `diff_profondeur=7` (dom a le double du banc) est un signal.
- `club_utilise_marge_100` : signal de sophistication tactique (le club
  exploite la marge de 100 pts Elo pour reordonner les echiquiers, A02 3.6.e).
  Rare mais potentiellement fort en interaction avec `decalage_position`.

**Fix** : tous en diff. `club_utilise_marge_100` en interaction avec
`decalage_position_blanc` (le club place strategiquement ET ce joueur est decale).

**Note win_rate_home_dom** : CatBoost SHAP = 0.075 (fort!) mais permutation = 0.
Probable redondance avec expected_score_recent (les deux captent la performance recente).
Le SHAP montre que le modele l'utilise dans ses arbres, mais shuffler la feature
n'affecte pas les predictions car d'autres features compensent.

### 4.3 CLUB HIERARCHY / VASES COMMUNIQUANTS (10/10 mortes)

`team_rank_in_club`, `club_nb_teams`, `reinforcement_rate`, `stabilite_effectif`,
`elo_moyen_evolution`, `joueur_promu/relegue`, `player_team_elo_gap`

**Le capitaine y pense ?** OUI. "On tire de l'equipe 2 pour renforcer l'equipe 1"
est une decision strategique majeure en interclubs multi-equipes.

**Pourquoi mortes ?**
- **Match-level** pour les features equipe (_dom/_ext)
- **Booleennes rares** pour les features joueur : `joueur_promu` est binaire
  et concerne ~5-10% des joueurs seulement
- `player_team_elo_gap` : redondant avec diff_elo (le modele voit deja la force relative)

**Fix** :
- `diff_reinforcement_rate` : l'une tire de ses reserves, pas l'autre
- `diff_stabilite_effectif` : equipe stable vs equipe en crise
- `joueur_promu × diff_elo` : un renforce face a un adversaire fort (interaction)
- `diff_elo_moyen_evolution` : une equipe se renforce, l'autre s'affaiblit

### 4.4 JOUEUR PROFIL (11/12 mortes)

`blanc_titre/noir_titre`, `elo_type`, `categorie`, `k_coefficient`, `data_quality`

**Le capitaine y pense ?** Peu. Le titre et la categorie sont deja dans l'Elo.

**Pourquoi mortes ?** Redondance avec Elo. Un GM a un Elo > 2400, un junior
a un K=40 mais son Elo reflete deja sa force.

**Exception: `k_coefficient`** — Le K indique la VOLATILITE de l'Elo.
Un junior K=40 a 1500 peut jouer 1300 ou 1700 selon le jour.
Un senior K=20 a 1500 est stable. Le diff_elo ne capte PAS cette incertitude.
**Feature manquante** : `elo_uncertainty = k_blanc + k_noir` (variance totale du matchup)
ou `k_diff = k_blanc - k_noir` (asymetrie de volatilite).

### 4.5 JOUEUR PRESENCE (11/12 mortes)

`taux_presence_saison/global`, `derniere_presence`, `regularite`,
`rondes_manquees_consecutives`, `joueur_fantome`

**Le capitaine y pense ?** OUI. Il sait qui est disponible et en rythme.

**Pourquoi mortes ?**
- `regularite` : std ~ 0 (3 valeurs, pas discriminant)
- `derniere_presence` : signal potentiel (rouille) mais pas differencie
- `joueur_fantome` : binaire rare

**Fix** :
- `diff_derniere_presence = derniere_presence_blanc - derniere_presence_noir`
  (un joueur en rythme vs un joueur rouille)
- `taux_presence_saison` : utiliser comme proxy de confiance du capitaine
  (un joueur aligne 9/10 rondes = titulaire indiscutable)

### 4.6 JOUEUR POSITION / ECHIQUIER (9/10 mortes)

`echiquier_moyen`, `echiquier_std`, `role_type`, `echiquier_prefere`, `flexibilite_echiquier`

**Le capitaine y pense ?** OUI. Il place chaque joueur sur l'echiquier optimal.

**Pourquoi mortes ?** `echiquier_moyen` et `echiquier_std` decrivent le joueur
en isolation. Le modele a deja `echiquier` (le board actuel) et `decalage_position`
(ecart entre board actuel et board habituel). Le `decalage_position` EST validated
car c'est implicitement relatif (position actuelle vs position habituelle).

**Fix** : `flexibilite_echiquier` pourrait interagir avec `decalage_position`
(un joueur flexible deplace souffre moins qu'un specialiste deplace).

### 4.7 COULEUR / WHITE-BLACK (10/14 mortes)

`win_rate_white/black`, `draw_rate_white/black`, `win_adv_white`, `draw_adv_white`,
`couleur_preferee`

**Le capitaine y pense ?** OUI. Convention FFE: echiquiers impairs = blancs
pour le dom. Le capitaine sait quelle couleur chaque board aura.

**Pourquoi mortes ?**
- Sparse : beaucoup de joueurs n'ont pas assez de matchs avec chaque couleur
- Pas d'interaction avec le board : `couleur_preferee=blanc` sans savoir
  si le joueur a effectivement les blancs sur ce board est inutile

**Feature manquante** : `color_match = (joueur a sa couleur preferee sur ce board)`
Calculable via la parite de l'echiquier + est_domicile + couleur_preferee.

### 4.8 PRESSION / CLUTCH (6 features, SHAP > 0 mais permutation negative)

`clutch_win/draw`, `pressure_type`, `win/draw_rate_pression`

**Le capitaine y pense ?** IMPLICITEMENT. Il sait que certains joueurs
"se transcendent" ou "craquent" sous pression.

**Pourquoi permutation negative ?** Les features de pression ont SHAP non-zero
(le modele les utilise dans ses arbres) mais permutation negative (les shuffler
AMELIORE le modele). Cela signifie que le modele **overfit** sur ces features.

**Diagnostic** : la definition de "match sous pression" (zone_enjeu IN montee/danger)
est peut-etre trop grossiere. 40% des matchs sont "danger", 13.5% "montee" = 54%
"sous pression". Trop frequent pour etre discriminant.

**Fix** : affiner la definition de pression. Par exemple :
- `pression_reelle = zone_danger AND ronde >= 7 AND ecart_dernier <= 2`
  (relegation reelle = fin de saison + proche de la chute)
- Ou : utiliser le differentiel `diff_clutch = clutch_blanc - clutch_noir`

---

## 5. Differentiels manquants

Chaque paire de features individuelles (_blanc/_noir ou _dom/_ext) devrait
avoir un DIFFERENTIEL correspondant. Le differentiel est plus informatif
car il capture le MATCHUP en 1 split au lieu de 2.

### 5.1 Differentiels joueur (board-level)

| Differentiel | Calcul | Signal metier |
|---|---|---|
| `diff_expected_score_recent` | ESR_blanc - ESR_noir | Avantage de forme |
| `diff_win_rate_recent` | WRR_blanc - WRR_noir | Momentum relatif |
| `diff_draw_rate` | DR_blanc - DR_noir | Profil nulles relatif |
| `diff_draw_rate_recent` | DRR_blanc - DRR_noir | Tendance nulles |
| `diff_win_rate_normal` | WRN_blanc - WRN_noir | Force de base relative |
| `diff_clutch_win` | clutch_blanc - clutch_noir | Performance sous pression |
| `diff_derniere_presence` | DP_blanc - DP_noir | Rouille relative |
| `diff_elo_trajectory` | ET_blanc - ET_noir | Progression relative |

### 5.2 Differentiels equipe (match-level, utilises via 1 split)

| Differentiel | Calcul | Signal metier |
|---|---|---|
| `diff_position` | position_dom - position_ext | Force classement relative |
| `diff_points_cumules` | PC_dom - PC_ext | Ecart saison |
| `diff_profondeur` | prof_dom - prof_ext | Banc relatif |
| `diff_stabilite_effectif` | stab_dom - stab_ext | Cohesion relative |
| `diff_rotation` | rot_dom - rot_ext | Experimentation relative |
| `diff_reinforcement_rate` | RR_dom - RR_ext | Crise relative |
| `diff_elo_moyen_evolution` | EME_dom - EME_ext | Renforcement relatif |
| `diff_win_rate_home` | WRH_dom - WRH_ext | Force domicile relative |
| `diff_draw_rate_home` | DRH_dom - DRH_ext | Profil domicile |

---

## 6. Interactions board x match

La SIGNATURE d'ALICE vs la litterature soccer : en interclubs, un match = 8 boards
avec le MEME contexte d'equipe mais des joueurs DIFFERENTS. Les interactions
capturent "comment CE joueur reagit a CE contexte d'equipe".

| Interaction | Calcul | Signal metier |
|---|---|---|
| `form_in_danger` | expected_score_recent_blanc * zone_danger_dom | Forme sous pression equipe |
| `decalage_important` | decalage_position_blanc * match_important | Deplace en match cle |
| `marge100_decale` | club_utilise_marge_100_dom * decalage_position_blanc | Strategie deliberee de placement |
| `draw_under_pressure` | draw_rate_blanc * zone_danger_dom | Joueur "nulle" en situation critique |
| `flex_decale` | flexibilite_echiquier_blanc * \|decalage_position_blanc\| | Joueur flexible deplace vs specialiste |
| `promu_vs_strong` | joueur_promu_blanc * max(0, -diff_elo) | Renforce face a adversaire fort |
| `color_match` | (couleur_preferee == couleur_effective) | Joueur a sa couleur preferee |

---

## 7. Features A CREER ou DERIVER

### 7.0 Synthese des sources par feature

| Feature proposee | Source litterature | Source ALICE (code existant) | Priorite |
|---|---|---|---|
| Differentiels joueur (§5.1) | NBA XGBoost PMC11265715 : "home values subtracted from away" | Paires _blanc/_noir dans recent_form.py, color_perf.py | **P0** |
| Differentiels equipe (§5.2) | Hubacek 2019 : pi-ratings relatifs | Paires _dom/_ext dans standings.py, club_behavior.py | **P0** |
| Home/away specialist | PMC8656876 (rugby), dashee87 (soccer) : individual home advantage | club_behavior.py a win_rate_home equipe, PAS joueur | **P1** |
| Elo uncertainty | Glicko-2 (Glickman 2001), arXiv:2512.18013 : K=learning rate bayesien | FEATURE_SPECIFICATION.md §3.2 : K-coefficient formule | **P1** |
| Color match | Convention FFE A02 : impairs=blancs pour dom | color_perf.py : couleur_preferee existe, composition.py : est_domicile | **P1** |
| Strength of schedule | arXiv:2410.21484 : "opponent strength adjustment", Hubacek pi-ratings | recent_form.py : win_rate_recent sans ajustement adversaire | **P2** |
| Interactions board×match (§6) | arXiv:1912.11762 : "richer features", PMC11265715 : SHAP interactions | pressure.py + standings.py + composition.py : composants existent | **P1** |
| Pression affinee | Bunker 2019 : contexte match > binaire | pressure.py : zone_enjeu IN (montee,danger) trop large (54%) | **P2** |
| Style joueur (aggressif/defensif) | NHSJS 2024 : game length, trades, queen lifetime | NON EXISTANT — necessite donnees de parties (PGN) | **P3** |

### 7.1 Home/away specialist joueur

**Litterature** : "Performance categories should be constructed separately for
home, away, and total performances" (systematic review arXiv:2410.21484).
En rugby, home advantage varie par equipe de +3% a +15% (PMC8656876).
En soccer, z-scores home/away + difference → 70% accuracy (Bryant University thesis).

**Code ALICE existant** : `club_behavior.py` calcule `win_rate_home` et
`draw_rate_home` mais au niveau EQUIPE, pas JOUEUR. `color_perf.py` calcule
les taux par COULEUR (blanc/noir pieces) qui est un proxy partiel (FFE convention:
dom = blancs sur echiquiers impairs) mais ne distingue pas domicile/exterieur.

**Ce qui manque** : taux de performance PAR JOUEUR quand son equipe joue a dom vs ext.

```python
# Decomposition dom/ext par joueur (nouveau)
joueur_games = filter(all_games, joueur in [blanc_nom, noir_nom])
home_games = filter(joueur_games, joueur.equipe == equipe_dom)
away_games = filter(joueur_games, joueur.equipe == equipe_ext)

pct_games_home_joueur = len(home_games) / len(joueur_games)
win_rate_home_joueur = wins(home_games) / len(home_games)
win_rate_away_joueur = wins(away_games) / len(away_games)
home_advantage_joueur = win_rate_home_joueur - win_rate_away_joueur
```

**Signaux** :
- `pct_games_home=0.85` : "home warrior" — le capitaine ne l'aligne qu'a domicile.
  Quand il joue a l'exterieur, il est hors zone de confort.
- `home_advantage_joueur > 0.15` : forte dependance au domicile — certains joueurs
  ont besoin de leur environnement familier (jeunes, joueurs anxieux).
- `home_advantage_joueur < -0.05` : "road warrior" — performe mieux a l'exterieur
  (moins de pression des supporteurs du club).

### 7.2 Elo uncertainty (derive du K-coefficient)

**Litterature** : Le systeme Glicko-2 (Glickman 2001) introduit un rating deviation (RD)
qui capture l'incertitude. "The K value reflects the Bayesian learning rate relevant
for each player, match situation, and skill context" (Springer s11257-025-09439).
arXiv:2512.18013 montre que K est un parametre empirique dont la valeur optimale
depend du contexte — pas une constante universelle.

**Code ALICE existant** : `FEATURE_SPECIFICATION.md §3.2` definit la formule
K-coefficient FIDE (K=40 juniors, K=20 adultes, K=10 elite >2400).
`categorie_blanc/noir` et `k_coefficient_blanc/noir` sont definis mais le K
n'est pas encore calcule dans le FE (statut: "A implementer").

**Ce qui manque** : le K comme feature de VOLATILITE, pas juste de categorie.

```python
# Volatilite Elo du matchup (nouveau)
elo_uncertainty = k_coefficient_blanc + k_coefficient_noir  # variance totale
k_asymmetry = k_coefficient_blanc - k_coefficient_noir      # asymetrie

# Ou derivation Glicko-like: RD = 350 * sqrt(1 + q^2 * rd_squared * sum_gi)
# Plus simple: RD_proxy = 350 / sqrt(nb_parties_jouees)
# Mais K est deja un bon proxy de RD
```

**Signaux** :
- `elo_uncertainty=80` (K40+K40) : deux juniors, Elos instables → outcome imprevisible,
  draw_rate probablement plus basse (juniors jouent "pour gagner")
- `elo_uncertainty=20` (K10+K10) : deux elites, Elos stables → Elo tres fiable,
  les features au-dela de l'Elo apportent peu
- `k_asymmetry=30` (K40 vs K10) : le junior a plus de variance que l'elite
  → le junior peut surprendre (upset) ou se faire ecraser (blowout)

### 7.3 Color match (derive convention FFE)

Convention FFE : echiquiers impairs = blancs pour le dom.
Le capitaine place deliberement les joueurs forts aux blancs sur les echiquiers
impairs a domicile.

```
couleur_effective = "blanc" if (est_domicile AND echiquier impair) else "noir"
color_match = (couleur_preferee == couleur_effective)
```

Signal : un joueur qui a sa couleur preferee a un avantage psychologique.

### 7.4 Strength of schedule (derive litterature multisports)

**Litterature** : "the strength of the opposition is a subtle aspect of feature
engineering — aggregated past performances should factor in the strength of opponents"
(review arXiv:1912.11762). En cricket, "rating systems adjusted for the strength
of opponents" ameliorent les predictions (arXiv:2410.21484). Le modele CAMP integre
"opponent strength, game situations, and player quality" dans un framework unifie.
En basketball NCAA, "strength of schedule" est un facteur cle du bracketology
(Georgia Tech 2026, bracketology driven by data).

**Code ALICE existant** : `recent_form.py` calcule `win_rate_recent` et
`expected_score_recent` sur les 5 derniers matchs SANS ponderation par force
adversaire. `standings.py` a `adversaire_niveau_dom/ext` (niveau hierarchique
de l'adversaire) mais c'est un niveau de competition, pas la force reelle.

**Ce qui manque** : ponderation de la forme par la force des adversaires rencontres.

```python
# Strength of schedule par joueur (nouveau)
recent_games = last_5_games(joueur, competition_level)
sos = mean(adversaire_elo for each game in recent_games)
sos_ratio = sos / mean_elo_competition_level  # > 1 = schedule difficile

# Forme ajustee
win_rate_adjusted = win_rate_recent * sos_ratio
# Ou: expected_score_adjusted = expected_score_recent * sos_ratio

# Alternative: decomposer forme vs adversaires forts/faibles
win_rate_vs_strong = wins(games WHERE adversaire_elo > joueur_elo + 100) / n
win_rate_vs_weak = wins(games WHERE adversaire_elo < joueur_elo - 100) / n
```

**Signaux** :
- `sos_ratio > 1.1` + `win_rate_recent > 0.6` : joueur sous-evalue (gagne contre des forts)
- `sos_ratio < 0.9` + `win_rate_recent > 0.7` : forme gonflée (gagne contre des faibles)
- `win_rate_vs_strong > 0.3` : "giant killer" (capable de faire tomber les favoris)
- `win_rate_vs_weak < 0.7` : instable contre les faibles (relachement)

---

## 8. References et sources

### Litterature multisports

- **Hubacek, Sourek & Zelezny 2019** — "Learning to predict soccer results from
  relational data with gradient boosted trees". Champion Soccer Prediction Challenge.
  CatBoost + pi-ratings, RPS=0.1925.
  https://link.springer.com/article/10.1007/s10994-018-5704-6

- **Berrar, Dubitzky et al. 2024** — "Evaluating soccer match prediction models:
  a deep learning approach and feature optimization for gradient-boosted trees".
  205 features, CatBoost best. arXiv:2309.14807

- **Review: Bunker & Thabtah 2019** — "The Application of Machine Learning
  Techniques for Predicting Results in Team Sport: A Review".
  "90% des papiers listent des features plus riches comme premier objectif".
  arXiv:1912.11762

- **Wilkens 2026** — "Can simple models predict football — and beat the odds?
  Lessons from the German Bundesliga". Isotonic calibration, ROI +10%.
  https://journals.sagepub.com/doi/10.1177/22150218261416681

- **Guo et al. 2017** — "On Calibration of Modern Neural Networks".
  Temperature scaling. Justifie init_score_alpha. arXiv:1706.04599

- **NBA XGBoost + SHAP (PMC11265715, 2024)** — "Features representing identical
  technical indicators for both teams are subtracted from each other".
  XGBoost AUC=0.982, 93.3% accuracy. SHAP: field goal % difference #1.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11265715/

- **Systematic Review ML Sports Betting (arXiv:2410.21484, 2024)** — 100+ papiers.
  "opponent strength adjustment" et "strength of schedule" parmi les features cles.
  Feature engineering > algorithm choice confirme.

- **Glickman 2001** — Systeme Glicko-2. Rating deviation (RD) = incertitude Elo.
  K-factor comme proxy de RD. Base theorique pour elo_uncertainty.

- **Maitra 2025** — "Empirical parameterization of the Elo Rating System".
  K optimal depend du contexte. arXiv:2512.18013

- **Rugby Home Advantage (PMC8656876, 2021)** — Home advantage varie +3% a +15%
  par equipe. Feature engineering dom/ext par equipe ET par joueur.

### Echecs ML

- **Pawnalyze 2022** — "Elo Ratings v. Machine Learning". LightGBM bat Elo,
  "the expected score alone from Elo's formulas is not enough".
  https://blog.pawnalyze.com/tournament/2022/02/27/Elo-Rating-Accuracy-Is-Machine-Learning-Better.html

- **MDPI Electronics 2025** — "ML Approaches for Classifying Chess Game Outcomes".
  Gradient boosting 83.2% accuracy, features beyond Elo.
  https://www.mdpi.com/2079-9292/15/1/1

- **Behavioral Programming Chess (arXiv:2504.05425, 2025)** — Strategies (central
  control, development, pawn structure) comme features. 80-83% accuracy vs Maia 47-53%.
  Applicable si donnees PGN disponibles (Phase 3+).

- **Chess Player Style (NHSJS 2024)** — 9 features de style (game length, trades,
  queen lifetime). Classification par style de GM. Signal : style joueur affecte
  prediction. NON applicable sans PGN.

- **Maia Chess (Microsoft/Toronto/Cornell)** — Predicts individual player moves
  with 75% accuracy. Players identifiable from 10 games at 86% accuracy.
  "Individual playing styles can be captured by ML." Implications : chaque joueur
  a un profil previsible que les features de forme/style captent partiellement.

### Reglements FFE

- **A02** : Championnat de France des Clubs (noyau 3.7.f, mutes 3.7.g, ordre Elo 3.6.e)
- **J02** : Competitions Jeunes (resultat_blanc=2.0 pour non-U10, section 4.1)
- **C01** : Coupe de France (pas de noyau)
- **FIDE 8.3.3** : K-coefficient (K=40 juniors, K=20 adultes, K=10 elite)
