#!/usr/bin/env python3
"""Module: generate_ml_graphs.py - ML Architecture Diagrams.

Document ID: ALICE-SCRIPTS-ML-GRAPHS-001
Version: 1.0.0

Genere des diagrams SVG de l'architecture ML Alice Engine:
- graphs/ml-pipeline.svg         : pipeline training V8 (residual learning)
- graphs/inference-chain.svg     : chaine inference production
- graphs/system-architecture.svg : architecture systeme 5 phases

ISO Compliance:
- ISO/IEC 42010 - Architecture (diagrams)
- ISO/IEC 42001 - AI Management (traceabilite pipeline ML)
- ISO/IEC 5055  - Code Quality (<300 lignes, SRP)

Usage:
    python -m scripts.generate_ml_graphs

Author: ALICE Engine Team
Last Updated: 2026-03-23
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from graphviz import Digraph
except ImportError:
    print("ERROR: graphviz package requis — pip install graphviz")
    sys.exit(1)

# Chemins
ROOT = Path(__file__).parent.parent
GRAPHS_DIR = ROOT / "graphs"

# Palette couleurs coherente
COLOR_DATA = "#E8F4FD"  # Sources de donnees — bleu clair
COLOR_PROCESS = "#E8FDE8"  # Traitement / FE — vert clair
COLOR_MODEL = "#FDE8D0"  # Modeles ML — orange clair
COLOR_OUTPUT = "#F0E8FD"  # Sorties / artefacts — violet clair
COLOR_EXTERNAL = "#F0F0F0"  # Services externes — gris clair
COLOR_GATE = "#FDEAEA"  # Quality gates — rouge clair


def _check_dot() -> bool:
    """Verifie que le binaire 'dot' (Graphviz) est sur le PATH. Retourne bool."""
    if shutil.which("dot") is None:
        print("  ! Graphviz (dot) absent du PATH.")
        print("    Windows: choco install graphviz | winget install graphviz")
        print("    Mac: brew install graphviz | Linux: apt install graphviz")
        return False
    return True


def _base_graph(name: str, label: str) -> Digraph:
    """Cree un Digraph avec attributs communs. Retourne Digraph configure."""
    g = Digraph(name, format="svg")
    g.attr(
        rankdir="TB",
        fontname="Helvetica",
        fontsize="12",
        bgcolor="transparent",
        label=label,
        labelloc="t",
        labeljust="c",
        pad="0.5",
        nodesep="0.5",
        ranksep="0.6",
    )
    g.attr("node", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="10")
    return g


def _node(g: Digraph, nid: str, label: str, color: str, shape: str = "box") -> None:
    """Ajoute un noeud stylise au graphe."""
    g.node(nid, label, shape=shape, style="rounded,filled", fillcolor=color, margin="0.2,0.1")


def _cluster_attr(sub: Digraph, label: str, fill: str, border: str) -> None:
    """Applique les attributs de cluster (label, style, couleurs)."""
    sub.attr(label=label, style="filled", fillcolor=fill, color=border)


def generate_ml_pipeline() -> bool:
    """Genere graphs/ml-pipeline.svg — pipeline training V8 residual learning.

    Returns True si generation reussie.
    """
    print("\n[1/3] Generation ml-pipeline.svg...")
    if not _check_dot():
        return False
    g = _base_graph("ml-pipeline", "V8 ML Training Pipeline — Residual Learning")

    _node(g, "parquet", "echiquiers.parquet\njoueurs.parquet", COLOR_DATA, "cylinder")
    _node(g, "fe_kernel", "FE Kernel 1\n(Kaggle P100) 196 cols", COLOR_PROCESS)
    _node(g, "splits", "train / valid / test\n(.parquet)", COLOR_DATA, "cylinder")
    _node(g, "baseline", "Elo Baseline\nbaselines.py", COLOR_PROCESS)
    _node(g, "draw_lookup", "draw_rate_lookup\n.parquet", COLOR_DATA, "cylinder")
    _node(g, "init_scores", "init_scores\n(log-odds)", COLOR_PROCESS)

    with g.subgraph(name="cluster_models") as m:
        m.attr(label="Modeles (Kaggle T4 GPU)", style="dashed", color="#AAAAAA")
        _node(m, "catboost", "CatBoost\nMultiClass", COLOR_MODEL)
        _node(m, "xgboost", "XGBoost\nmulti:softprob", COLOR_MODEL)
        _node(m, "lgbm", "LightGBM\nmulticlass", COLOR_MODEL)

    _node(g, "calib", "Calibration\nIsotonic / classe", COLOR_PROCESS)
    _node(g, "gate", "Quality Gate\n9 conditions\n(recall_draw > 1%)", COLOR_GATE, "diamond")
    _node(g, "artifacts", "Artefacts\n.cbm/.ubj/.txt\n+ calibrators", COLOR_OUTPUT)
    _node(g, "hf_hub", "HF Hub\nPierrax/alice-engine", COLOR_EXTERNAL)

    for src, dst in [
        ("parquet", "fe_kernel"),
        ("fe_kernel", "splits"),
        ("splits", "baseline"),
        ("baseline", "draw_lookup"),
        ("baseline", "init_scores"),
        ("draw_lookup", "init_scores"),
        ("splits", "catboost"),
        ("splits", "xgboost"),
        ("splits", "lgbm"),
        ("catboost", "calib"),
        ("xgboost", "calib"),
        ("lgbm", "calib"),
        ("calib", "gate"),
        ("artifacts", "hf_hub"),
    ]:
        g.edge(src, dst)
    g.edge("init_scores", "catboost", label="base_margin")
    g.edge("init_scores", "xgboost", label="base_margin")
    g.edge("init_scores", "lgbm", label="init_score")
    g.edge("gate", "artifacts", label="PASS")

    g.render(str(GRAPHS_DIR / "ml-pipeline"), cleanup=True)
    print("  OK graphs/ml-pipeline.svg")
    return True


def generate_inference_chain() -> bool:
    """Genere graphs/inference-chain.svg — chaine inference production.

    Returns True si generation reussie.
    """
    print("\n[2/3] Generation inference-chain.svg...")
    if not _check_dot():
        return False
    g = _base_graph("inference-chain", "Production Inference Chain — Alice Engine")

    _node(
        g,
        "api_req",
        "API Request\nPOST /api/v1/predict\n(joueur, adversaire, contexte)",
        COLOR_EXTERNAL,
    )
    _node(g, "feat_store", "Feature Store\n(joueur + equipe features)", COLOR_DATA, "cylinder")
    _node(g, "elo_baseline", "Elo Baseline\nbaselines.py", COLOR_PROCESS)
    _node(g, "draw_lookup2", "draw_rate_lookup\n.parquet", COLOR_DATA, "cylinder")
    _node(g, "init_scores2", "init_scores\n(log-odds par board)", COLOR_PROCESS)

    with g.subgraph(name="cluster_pred") as p:
        p.attr(label="Modele CatBoost (RAM)", style="dashed", color="#AAAAAA")
        _node(p, "predict", "predict_with_init()\npredict_proba", COLOR_MODEL)

    _node(g, "calib2", "Calibration\nIsotonic / classe", COLOR_PROCESS)
    _node(g, "proba", "P(Win) / P(Draw) / P(Loss)\npar board", COLOR_OUTPUT)
    _node(g, "ce", "CE Optimizer\nE[score] = P(win) + 0.5*P(draw)", COLOR_PROCESS)
    _node(g, "compo", "Composition\nRecommandee + alternatives", COLOR_OUTPUT)

    for src, dst in [
        ("api_req", "feat_store"),
        ("api_req", "elo_baseline"),
        ("elo_baseline", "draw_lookup2"),
        ("elo_baseline", "init_scores2"),
        ("draw_lookup2", "init_scores2"),
        ("feat_store", "predict"),
        ("predict", "calib2"),
        ("calib2", "proba"),
        ("proba", "ce"),
        ("ce", "compo"),
    ]:
        g.edge(src, dst)
    g.edge("init_scores2", "predict", label="base_margin")

    g.render(str(GRAPHS_DIR / "inference-chain"), cleanup=True)
    print("  OK graphs/inference-chain.svg")
    return True


def generate_system_architecture() -> bool:
    """Genere graphs/system-architecture.svg — architecture 5 phases.

    Returns True si generation reussie.
    """
    print("\n[3/3] Generation system-architecture.svg...")
    if not _check_dot():
        return False
    g = _base_graph("system-architecture", "Alice Engine — Architecture Systeme (5 phases)")

    with g.subgraph(name="cluster_sources") as s:
        s.attr(label="Sources de donnees", style="filled", fillcolor="#F8FBFF", color="#AAAACC")
        _node(s, "ffe_scrapper", "FFE Scrapper\n(ffe-history repo)", COLOR_EXTERNAL)
        _node(s, "hf_dataset", "HF Hub\nPierrax/ffe-history", COLOR_EXTERNAL)
        _node(s, "hf_model", "HF Hub\nPierrax/alice-engine", COLOR_EXTERNAL)

    _node(g, "parquet2", "echiquiers.parquet\njoueurs.parquet", COLOR_DATA, "cylinder")

    with g.subgraph(name="cluster_p1") as p1:
        _cluster_attr(p1, "Phase 1 — V8 ML (Training)", "#FFF8F0", "#CCAAAA")
        _node(p1, "fe", "FE Kernel 1 / 196 features", COLOR_PROCESS)
        _node(p1, "ml", "CatBoost / XGB / LGBM\nResidual Learning", COLOR_MODEL)
        _node(p1, "qg", "Quality Gate / 9 conditions", COLOR_GATE, "diamond")

    with g.subgraph(name="cluster_p2") as p2:
        _cluster_attr(p2, "Phase 2 — Feature Store + API", "#F0FFF0", "#AACCAA")
        _node(p2, "fstore", "Feature Store\n(parquets pre-calcules)", COLOR_DATA, "cylinder")
        _node(p2, "api2", "FastAPI\nPOST /api/v1/predict", COLOR_PROCESS)

    with g.subgraph(name="cluster_p3") as p3:
        _cluster_attr(p3, "Phase 3 — ALI (Adversarial Lineup Inference)", "#F0F8FF", "#AAAACC")
        _node(p3, "ali", "ALI / 20 scenarios adversaire", COLOR_PROCESS)

    with g.subgraph(name="cluster_p4") as p4:
        _cluster_attr(p4, "Phase 4 — CE V9 (Composition Engine)", "#FFF0FF", "#CCAACC")
        _node(p4, "ce2", "OR-Tools / Multi-equipe", COLOR_PROCESS)
        _node(p4, "out", "Composition optimale\n+ E[score]", COLOR_OUTPUT)

    with g.subgraph(name="cluster_p5") as p5:
        _cluster_attr(p5, "Phase 5 — Deploy (Oracle VM)", "#F8F8F0", "#CCCCAA")
        _node(p5, "oracle", "Oracle Always Free\n4 OCPUs ARM / 24 GB RAM", COLOR_EXTERNAL)
        _node(p5, "vercel", "Vercel (chess-app)\nNext.js frontend", COLOR_EXTERNAL)

    for src, dst in [
        ("ffe_scrapper", "parquet2"),
        ("hf_dataset", "parquet2"),
        ("parquet2", "fe"),
        ("fe", "ml"),
        ("ml", "qg"),
        ("hf_model", "fstore"),
        ("fstore", "api2"),
        ("parquet2", "fstore"),
        ("api2", "ali"),
        ("api2", "ce2"),
        ("ce2", "out"),
        ("api2", "oracle"),
        ("ce2", "out"),
    ]:
        g.edge(src, dst)
    g.edge("qg", "hf_model", label="PASS")
    g.edge("ali", "ce2", label="20 scenarios")
    g.edge("oracle", "vercel", label="HTTPS")

    g.render(str(GRAPHS_DIR / "system-architecture"), cleanup=True)
    print("  OK graphs/system-architecture.svg")
    return True


def main() -> int:
    """Genere les 3 diagrams SVG de l'architecture ML. Retourne 0/1."""
    print(f"\n{'=' * 60}\n  GENERATION DIAGRAMS ML ARCHITECTURE\n{'=' * 60}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Output: {GRAPHS_DIR}")

    GRAPHS_DIR.mkdir(exist_ok=True)

    results = [
        ("ML Pipeline", generate_ml_pipeline()),
        ("Inference Chain", generate_inference_chain()),
        ("System Architecture", generate_system_architecture()),
    ]

    print(f"\n{'=' * 60}\n  RESUME\n{'=' * 60}")
    success = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  [{'OK' if ok else '!'}] {name}")
    print(f"\nScore: {success}/{total}")

    if success == total:
        print("\nTous les diagrams ML generes avec succes!")
        return 0
    print("\nErreur: certains diagrams n'ont pas pu etre generes.")
    print("Verifier: pip install graphviz  +  choco/brew/apt install graphviz")
    return 1


if __name__ == "__main__":
    sys.exit(main())
