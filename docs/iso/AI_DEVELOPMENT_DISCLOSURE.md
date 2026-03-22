# AI-Assisted Development Disclosure

**Document ID:** ALICE-ISO42001-AI-DISCLOSURE
**Version:** 1.0.0
**Date:** 2026-03-22
**Norme:** ISO/IEC 42001:2023 Clause 6.1.2 (AI system transparency)

---

## 1. Statement

ALICE Engine is developed with AI assistance from Anthropic's Claude Code CLI.
The human developer (Pierre Guillemin) retains full authorship, review authority,
and responsibility for all code, architecture decisions, and documentation.

## 2. AI Tool

| Attribute | Value |
|-----------|-------|
| Tool | Claude Code (Anthropic CLI) |
| Models used | Claude Opus 4.5, Opus 4.6, Sonnet 4.6 |
| Role | Pair programming, code generation, documentation, review |
| Authority | Advisory only — human approves every commit |

## 3. Scope of AI Contribution

| Activity | AI Role | Human Role |
|----------|---------|------------|
| Architecture decisions | Proposes options | Decides |
| Feature engineering | Implements specs | Writes specs, validates domain logic |
| Training pipeline | Writes code | Reviews, validates metrics |
| ISO documentation | Drafts content | Reviews, approves |
| Bug fixes | Diagnoses, proposes fix | Confirms root cause, approves |
| Code review | Reviews code | Final authority |
| Kaggle deployment | Pre-flight checks, scripts | Push decision, monitoring |

## 4. Traceability

Every AI-assisted commit includes a `Co-Authored-By` trailer in the commit body:

```
Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

### Statistics (as of 2026-03-22)

| Metric | Value |
|--------|-------|
| Total commits | 299 |
| AI co-authored | 269 (90%) |
| Human-only | 30 (10%) |

### Model breakdown

| Model | Commits |
|-------|---------|
| Claude Opus 4.5 | 158 |
| Claude Opus 4.6 (1M context) | 53 |
| Claude Opus 4.6 | 35 |
| Claude Sonnet 4.6 | 23 |

To audit AI contribution on any file:

```bash
git log --all --format='%H %s' -- path/to/file.py | while read hash msg; do
  git log -1 --format='%b' $hash | grep -q "Co-Authored" && echo "AI: $msg" || echo "Human: $msg"
done
```

## 5. Quality Assurance

AI-generated code passes the same quality gates as human code:

- **Pre-commit:** Gitleaks, Ruff lint+format, MyPy, Bandit
- **Commit-msg:** Commitizen conventional commits
- **Pre-push:** Pytest >80% coverage, Xenon complexity B, pip-audit, architecture analysis
- **Code review:** Human reviews all diffs before commit approval

No AI-generated code bypasses these checks.

## 6. Limitations Acknowledged

- AI does not have access to production systems or user data
- AI cannot push to remote repositories without human approval
- AI may produce plausible but incorrect domain logic (chess rules, FFE regulations)
  — all domain logic is validated against official FFE documentation
- AI-generated ISO documentation requires human review for normative accuracy

## 7. References

- ISO/IEC 42001:2023 Clause 6.1.2 — Risk assessment for AI systems
- ISO/IEC 42001:2023 Clause 7.5 — Documented information
- EU AI Act Article 52 — Transparency obligations
- Anthropic Usage Policy — https://www.anthropic.com/policies
