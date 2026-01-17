# ISO Pipeline Hooks - Documentation Technique

**Document ID:** ALICE-DOC-HOOKS-001
**Version:** 1.0.0
**Date:** 2026-01-17
**Classification:** Outil de Développement
**Normes:** ISO/IEC 42001, 24029, 24027, 5055, 5259, 25059, 42005

---

## 1. Objectif

Ce document décrit le système de hooks Claude Code implémentant une **boucle auto-corrective ISO** pour le pipeline d'entraînement ML ALICE Engine.

### 1.1 Problème Résolu

| Problème | Solution |
|----------|----------|
| Étapes pipeline oubliées | Hook `UserPromptSubmit` injecte l'état dans le contexte |
| Prérequis non vérifiés | Hook `PreToolUse` bloque et guide les corrections |
| Seuils ISO non validés | Hook `PostToolUse` vérifie les thresholds |
| État non persistant | `pipeline_state.json` trace l'avancement |
| Contournement par renommage | Vérification **filesystem**, pas patterns commandes |

### 1.2 Conformité ISO

| Norme | Application |
|-------|-------------|
| ISO/IEC 42001:2023 | Traçabilité pipeline, Model Card |
| ISO/IEC 24029:2021 | Validation robustesse (noise_tolerance ≥ 0.95) |
| ISO/IEC TR 24027:2021 | Validation équité (demographic_parity ≥ 0.60) |
| ISO/IEC 5055:2021 | Qualité code hooks (< 300 lignes/fichier) |
| ISO/IEC 5259:2024 | Data lineage, validation schéma |
| ISO/IEC 25059:2023 | Rapport qualité AI final |
| ISO/IEC 42005:2025 | Impact Assessment |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOUCLE AUTO-CORRECTIVE ISO                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ UserPromptSubmit │ → inject_pipeline.py                      │
│  └────────┬─────────┘   Injecte état pipeline dans contexte     │
│           │              Claude voit: "NEXT STEP: X"            │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │   PreToolUse     │ → pre_check.py                            │
│  └────────┬─────────┘   Vérifie prérequis AVANT exécution       │
│           │              BLOCK si manquants + instructions FIX   │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │   Exécution      │   Commande Bash (training, validation)    │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │   PostToolUse    │ → pipeline_gate.py                        │
│  └────────┬─────────┘   Vérifie ISO thresholds APRÈS exécution  │
│           │              Update pipeline_state.json              │
│           │              LOOP si non-conforme                    │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │      Stop        │ → stop_summary.py                         │
│  └──────────────────┘   Résumé final avant fin réponse          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Fichiers

### 3.1 Structure

```
.claude/
├── settings.json              # Configuration hooks Claude Code
└── hooks/
    ├── pre_check.py           # PreToolUse - Bloque si prérequis manquants
    ├── post_validate.py       # PostToolUse - Valide ISO thresholds
    ├── pipeline_gate.py       # PostToolUse - Update état pipeline
    ├── inject_pipeline.py     # UserPromptSubmit - Injecte status
    ├── stop_summary.py        # Stop - Résumé final
    └── pipeline_state.json    # État persistant du pipeline
```

### 3.2 Configuration (settings.json)

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "python .claude/hooks/pre_check.py",
          "timeout": 30
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "python .claude/hooks/pipeline_gate.py",
          "timeout": 30
        }]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python .claude/hooks/inject_pipeline.py",
          "timeout": 10
        }]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python .claude/hooks/stop_summary.py",
          "timeout": 10
        }]
      }
    ]
  }
}
```

---

## 4. Pipeline ISO

### 4.1 Étapes

| # | Étape | Norme ISO | Prérequis | Vérification |
|---|-------|-----------|-----------|--------------|
| 1 | Data Preparation | 5259 | - | `data/features/train.parquet` EXISTS |
| 2 | Baseline Models | 25059 | 1 | `models/v*/metadata.json` EXISTS |
| 3 | AutoGluon Training | 42001 | 1 | `models/autogluon/*/predictor.pkl` EXISTS |
| 4 | Robustness Validation | 24029 | 3 | `noise_tolerance >= 0.95` |
| 5 | Fairness Validation | 24027 | 3 | `demographic_parity >= 0.60` |
| 6 | McNemar Comparison | 24029 | 2, 3 | `reports/mcnemar_comparison.json` EXISTS |
| 7 | Model Card | 42001 | 3 | `reports/iso42001_model_card.json` EXISTS |
| 8 | Impact Assessment | 42005 | 3 | `reports/iso42005_impact_assessment.json` EXISTS |
| 9 | Final Report | 25059 | 4-8 | `reports/ISO_25059_TRAINING_REPORT_*.md` EXISTS |

### 4.2 Graphe de Dépendances

```
                    ┌─────────────────┐
                    │ 1. Data Ready   │
                    │    (ISO 5259)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              │
     ┌─────────────┐  ┌─────────────┐       │
     │ 2. Baseline │  │ 3. AutoGluon│       │
     │ (ISO 25059) │  │ (ISO 42001) │       │
     └──────┬──────┘  └──────┬──────┘       │
            │                │              │
            │    ┌───────────┼───────────┐  │
            │    ▼           ▼           ▼  │
            │ ┌────────┐ ┌────────┐ ┌────────┐
            │ │4.Robust│ │5.Fair  │ │7.Card  │
            │ │(24029) │ │(24027) │ │(42001) │
            │ └───┬────┘ └───┬────┘ └───┬────┘
            │     │          │          │
            │     └────┬─────┴──────────┤
            │          ▼                │
            │    ┌───────────┐          │
            └───►│6.McNemar  │          │
                 │ (24029)   │          │
                 └─────┬─────┘          │
                       │    ┌───────────┘
                       ▼    ▼
                 ┌───────────────┐
                 │ 9. Final Rpt  │
                 │  (ISO 25059)  │
                 └───────────────┘
```

### 4.3 Seuils ISO

| Métrique | Seuil Minimum | Seuil Idéal | Norme |
|----------|---------------|-------------|-------|
| Test AUC | 0.70 | 0.75 | ISO 25059 |
| Noise Tolerance | 0.95 | 1.00 | ISO 24029 |
| Demographic Parity | 0.60 (CAUTION) | 0.80 (FAIR) | ISO 24027 |
| Coverage Code | 70% | 80% | ISO 29119 |

---

## 5. Fonctionnement des Hooks

### 5.1 PreToolUse (pre_check.py)

**Déclencheur:** Avant chaque commande Bash

**Vérifications:**
- Training: data files + RAM ≥ 6GB
- Robustness: AutoGluon model + data
- Fairness: AutoGluon model
- McNemar: AutoGluon + Baseline models
- Final Report: Tous rapports intermédiaires

**Sortie si échec:**
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "MISSING: data/features/train.parquet",
    "additionalContext": "BLOCKED - Prerequisites missing:\n  - MISSING: data\nFIX:\n  - Run feature engineering first"
  }
}
```

**Code de sortie:** `exit 2` = BLOCK

### 5.2 PostToolUse (pipeline_gate.py)

**Déclencheur:** Après chaque commande Bash

**Actions:**
1. Scan filesystem pour vérifier état réel
2. Compare avec `pipeline_state.json`
3. Update état si étape complétée
4. Output `additionalContext` si non-conforme

**Vérification ISO:**
```python
# Vérifie contenu JSON, pas juste existence
_check_json_condition(cwd, "reports/iso24029_robustness.json EXISTS AND noise_tolerance >= 0.95")
```

### 5.3 UserPromptSubmit (inject_pipeline.py)

**Déclencheur:** Chaque prompt utilisateur

**Sortie injectée dans contexte Claude:**
```
## ISO PIPELINE STATUS (MANDATORY)
[x] Data Preparation (ISO 5259)
[x] AutoGluon Training (ISO 42001)
[ ] Robustness Validation (ISO 24029)

**NEXT REQUIRED STEP: Robustness Validation**
You MUST complete this step before proceeding.
```

**Effet:** Claude ne peut pas ignorer car injecté dans son contexte système.

### 5.4 Stop (stop_summary.py)

**Déclencheur:** Fin de chaque réponse Claude

**Sortie:**
- `ISO COMPLETE` si tous rapports présents
- `ISO INCOMPLETE - Missing: X, Y` sinon

---

## 6. État Pipeline (pipeline_state.json)

### 6.1 Structure

```json
{
  "version": "1.0.0",
  "last_updated": "2026-01-17T14:00:00",
  "steps": {
    "step_id": {
      "name": "Human readable name",
      "iso": "ISO XXXX",
      "requires": ["prerequisite_step_ids"],
      "check": "path/file EXISTS AND field >= value",
      "completed": true,
      "timestamp": "2026-01-17T04:25:45",
      "result": {"metric": 0.95, "status": "PASS"}
    }
  },
  "summary": {
    "total_steps": 9,
    "completed": 9,
    "status": "COMPLETE",
    "warnings": ["Fairness at 64.84% < 80% target"]
  }
}
```

### 6.2 Syntaxe des Checks

| Syntaxe | Exemple | Signification |
|---------|---------|---------------|
| `EXISTS` | `file.json EXISTS` | Fichier existe |
| `* wildcard` | `models/v*/metadata.json EXISTS` | Pattern glob |
| `AND condition` | `file.json EXISTS AND field >= 0.95` | Existence + valeur JSON |

---

## 7. Commandes Pipeline

### 7.1 Exécution Manuelle (Étape par Étape)

```bash
# Étape 3: Entraînement AutoGluon
python -m scripts.autogluon.run_training

# Étape 4: Validation Robustesse
python -m scripts.autogluon.iso_robustness

# Étape 5: Validation Équité
python -m scripts.autogluon.iso_fairness

# Étape 6: Comparaison McNemar
python -m scripts.comparison.run_mcnemar

# Étape 9: Rapport Final
python -m scripts.reports.generate_iso25059
```

### 7.2 Vérification État

```bash
# Afficher état pipeline
cat .claude/hooks/pipeline_state.json | python -m json.tool

# Vérifier rapports générés
ls -la reports/*.json reports/*.md
```

---

## 8. Dépannage

### 8.1 Hook ne se déclenche pas

```bash
# Redémarrer session Claude Code
/exit
claude
```

### 8.2 Chemin hook incorrect

```bash
# Vérifier settings.json
cat .claude/settings.json

# Tester hook manuellement
echo '{"tool_name":"Bash","tool_input":{"command":"test"},"cwd":"C:/Dev/Alice-Engine"}' | python .claude/hooks/pre_check.py
```

### 8.3 Reset état pipeline

```bash
# Supprimer état pour re-scan
rm .claude/hooks/pipeline_state.json

# Prochain PostToolUse recréera l'état
```

---

## 9. Conformité ISO 5055

| Fichier | Lignes | Limite | Status |
|---------|--------|--------|--------|
| pre_check.py | 122 | 300 | ✅ |
| post_validate.py | 99 | 300 | ✅ |
| pipeline_gate.py | 135 | 300 | ✅ |
| inject_pipeline.py | 89 | 300 | ✅ |
| stop_summary.py | 58 | 300 | ✅ |

---

## 10. Changelog

| Version | Date | Changements |
|---------|------|-------------|
| 1.0.0 | 2026-01-17 | Initial release |

---

## 11. Références

- [Claude Code Hooks Documentation](https://docs.anthropic.com/claude-code/hooks)
- [ISO/IEC 42001:2023 - AI Management System](https://www.iso.org/standard/81230.html)
- [ISO/IEC 24029:2021 - Neural Network Robustness](https://www.iso.org/standard/77609.html)
- [ISO/IEC TR 24027:2021 - Bias in AI](https://www.iso.org/standard/77607.html)

---

**Document Control**

| Rôle | Nom |
|------|-----|
| Auteur | Claude Opus 4.5 |
| Validateur | - |
| Approbateur | - |
