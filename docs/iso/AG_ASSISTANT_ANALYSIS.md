# AutoGluon-Assistant (MLZero) - Analyse et Adaptation ALICE

**Document ID**: ALICE-AG-ANALYSIS-001
**Version**: 1.0.0
**Date**: 2026-01-17
**ISO Compliance**: ISO/IEC 42001:2023 (AI Management)

---

## 1. Vue d'ensemble

AutoGluon-Assistant (AG-A), également connu sous le nom **MLZero**, est un système multi-agent alimenté par LLM qui automatise les workflows ML de bout en bout sans intervention humaine.

### Références
- [GitHub Repository](https://github.com/autogluon/autogluon-assistant)
- [Amazon Science Blog](https://www.amazon.science/blog/autogluon-assistant-zero-code-automl-through-multiagent-collaboration)
- [Project MLZero](https://project-mlzero.github.io/)
- Paper: "MLZero: A Multi-Agent System for End-to-end ML Automation" (NeurIPS 2025)

---

## 2. Architecture Multi-Agent

AG-A comprend **4 modules spécialisés**, chacun étant un sous-système avec un ou plusieurs agents LLM:

### 2.1 Perception Module
**Fonction**: Interprète les entrées de données arbitraires et les transforme en contexte structuré.

**Capacités**:
- Parsing de fichiers CSV avec variables cibles floues
- Analyse des distributions de colonnes
- Inférence de la structure de tâche

**Adaptation ALICE**:
```python
# Le module perception d'ALICE pourrait analyser:
# - Fichiers parquet FFE (historique parties)
# - Structure des compétitions (nationale vs régionale)
# - Distribution ELO par ligue
```

### 2.2 Semantic Memory Module
**Fonction**: Enrichit le système avec la connaissance des bibliothèques ML.

**Capacités**:
- Sélection automatique des outils appropriés
- Mapping tâches → modèles
- Best practices AutoGluon

**Adaptation ALICE**:
```python
# Semantic memory pour ALICE:
# - Connaissance des règles FFE
# - Contraintes de composition d'équipe
# - Seuils ISO (fairness, robustness)
```

### 2.3 Episodic Memory Module
**Fonction**: Maintient l'historique chronologique des exécutions pour debugging ciblé.

**Capacités**:
- Tracking succès/échecs
- Contexte de debugging
- Support du raffinement itératif

**Adaptation ALICE**:
```python
# Episodic memory pour ALICE:
# - Historique des entraînements (MLflow)
# - Échecs de validation ISO
# - Patterns de non-conformité
```

### 2.4 Iterative Coding Module
**Fonction**: Implémente un processus de raffinement avec boucles de feedback.

**Capacités**:
- Exécution de code
- Analyse des erreurs
- Injection de connaissance experte

**Adaptation ALICE**:
```python
# Iterative coding pour ALICE:
# - Génération de code d'entraînement
# - Correction automatique des échecs
# - Adaptation aux contraintes ISO
```

---

## 3. Performance Benchmark

| Benchmark | Score AG-A | Compétiteurs |
|-----------|------------|--------------|
| MAAB (25 tâches) | 92% succès | - |
| MLE-bench Lite | 86% succès, Rank 1.43 | AIDE: 2.36, ResearchAgent: 3.29 |
| Kaggle AutoML Grand Prix 2024 | 10ème place | Seul agent automatisé à scorer |
| Modèle 8B compact | 45.3% succès | - |

---

## 4. Intégration MCP (Model Context Protocol)

AG-A supporte l'intégration MCP pour orchestration distante:

```yaml
# Configuration MCP pour ALICE
mcp_config:
  provider: "anthropic"  # Claude
  tools:
    - train_model
    - validate_iso
    - deploy_render
```

**Avantage**: Permet à Claude Code d'orchestrer directement le pipeline ALICE via MCP.

---

## 5. Analyse des 5 Non-Conformités Potentielles

### 5.1 NC-001: Absence de Perception Automatique
**Problème**: Le pipeline ALICE actuel nécessite une spécification manuelle des features.
**Solution AG-A**: Implémenter un module de perception qui analyse automatiquement les fichiers parquet.

### 5.2 NC-002: Pas de Mémoire Sémantique ISO
**Problème**: Les seuils ISO (fairness 0.8, robustness 0.95) sont hardcodés.
**Solution AG-A**: Créer une base de connaissance des normes ISO avec règles de décision.

### 5.3 NC-003: Debugging Manuel des Échecs
**Problème**: Les échecs de validation nécessitent une analyse manuelle.
**Solution AG-A**: Implémenter un episodic memory qui trace et explique les échecs.

### 5.4 NC-004: Pas de Raffinement Automatique
**Problème**: En cas d'échec fairness, l'action corrective est manuelle.
**Solution AG-A**: Ajouter un module de raffinement itératif avec actions correctives automatiques.

### 5.5 NC-005: Absence d'Orchestration LLM
**Problème**: Le pipeline n'est pas pilotable par LLM.
**Solution AG-A**: Intégrer MCP pour permettre l'orchestration par Claude ou autre LLM.

---

## 6. Plan d'Adaptation ALICE

### Phase 1: Mémoire Sémantique ISO (Priorité Haute)
```python
# scripts/agents/semantic_memory.py
ISO_KNOWLEDGE_BASE = {
    "fairness": {
        "standard": "ISO/IEC TR 24027:2021",
        "thresholds": {"demographic_parity": 0.8, "critical": 0.6},
        "mitigations": ["reweighting", "resampling", "constraint"],
    },
    "robustness": {
        "standard": "ISO/IEC 24029:2021",
        "thresholds": {"noise_tolerance": 0.95, "consistency": 0.9},
        "tests": ["noise", "dropout", "monotonicity"],
    },
}
```

### Phase 2: Mémoire Épisodique MLflow (Priorité Moyenne)
```python
# scripts/agents/episodic_memory.py
class EpisodicMemory:
    def __init__(self, mlflow_uri: str):
        self.client = MlflowClient(mlflow_uri)

    def get_failure_patterns(self, metric: str) -> list[dict]:
        """Récupère les patterns d'échec pour un métrique ISO."""
        runs = self.client.search_runs(
            filter_string=f"metrics.{metric} < 0.8",
            order_by=["start_time DESC"],
        )
        return [self._analyze_failure(r) for r in runs]
```

### Phase 3: Raffinement Itératif (Priorité Haute)
```python
# scripts/agents/iterative_refinement.py
class IterativeRefinement:
    def refine_fairness(self, report: ISO24027EnhancedReport) -> dict:
        """Raffine automatiquement en cas d'échec fairness."""
        if report.status == "CRITICAL":
            return self._apply_reweighting(report.group_analyses)
        elif report.status == "CAUTION":
            return self._schedule_review(report)
        return {"action": "none", "reason": "compliant"}
```

### Phase 4: Intégration MCP (Priorité Basse)
```yaml
# .claude/mcp_tools.yaml
tools:
  - name: alice_train
    description: "Train ALICE model with ISO validation"
    parameters:
      model_type: ["autogluon", "catboost", "xgboost", "lightgbm"]
      validate_iso: true

  - name: alice_validate
    description: "Run ISO validation on trained model"
    parameters:
      model_path: string
      standards: ["24027", "24029", "42005"]
```

---

## 7. Installation AG-A (Référence)

```bash
# Prérequis: Linux, Conda, Python 3.10-3.12
pip install uv
uv pip install git+https://github.com/autogluon/autogluon-assistant.git

# Configuration AWS Bedrock (défaut)
export AWS_DEFAULT_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="<key>"
export AWS_SECRET_ACCESS_KEY="<secret>"

# Usage
mlzero -i <input_data_folder>
```

---

## 8. Recommandations

### Court terme (Sprint actuel)
1. **Implémenter la mémoire sémantique ISO** dans `scripts/agents/`
2. **Créer le module de raffinement itératif** pour corrections automatiques fairness

### Moyen terme (Prochain sprint)
3. **Intégrer MLflow comme mémoire épisodique** avec analyse des patterns d'échec
4. **Ajouter un module de perception** pour analyse automatique des données FFE

### Long terme
5. **Intégration MCP complète** pour orchestration par Claude Code
6. **Évaluation sur benchmark MAAB** pour validation de l'architecture

---

## 9. Conclusion

AutoGluon-Assistant offre une architecture multi-agent robuste qui peut résoudre les 5 non-conformités identifiées. L'adaptation progressive (mémoire sémantique → raffinement → épisodique → MCP) permet une intégration incrémentale sans disruption du pipeline existant.

**Priorité immédiate**: Implémenter la mémoire sémantique ISO et le raffinement itératif pour automatiser les corrections de fairness.

---

## Sources

- [AutoGluon-Assistant GitHub](https://github.com/autogluon/autogluon-assistant)
- [Amazon Science Blog - Zero-code AutoML](https://www.amazon.science/blog/autogluon-assistant-zero-code-automl-through-multiagent-collaboration)
- [Project MLZero](https://project-mlzero.github.io/)
- [NeurIPS 2025 Paper](https://openreview.net/pdf/55f28109c8ee532fe1c950142c23f6efd636a79e.pdf)
