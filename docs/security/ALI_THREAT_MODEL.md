# ALI Threat Model — STRIDE Analysis

**Document ID** : ALICE-THREAT-MODEL-ALI-v1.0.0
**Standards** : Microsoft STRIDE, ISO/IEC 27001:2022 (ISMS), ISO/IEC 27034:2011
(Application Security), ISO/IEC 25059:2023 (AI System Quality §Security),
OWASP API Security Top 10 (2023)
**Scope** : ALI prediction service `/api/v1/compose` + `/api/v1/recompose` +
admin/audit endpoints
**Generated** : 2026-04-28
**Status** : DRAFT (T20 Plan 3 V2 §G5 fix-on-sight). Full pen-test deferred Phase 5.

---

## 1. System Boundary

### 1.1 Trust Boundaries

```
[ Public Internet ]
    │
    ▼ HTTPS (TLS 1.3, Phase 5)
[ Reverse Proxy (Phase 5 : Caddy/Nginx + slowapi rate limit) ]
    │
    ▼ HTTP localhost
[ FastAPI app : Pydantic validation + JWT auth (Phase 5) ]
    │
    ▼ in-process
[ Service Layer : ScenarioGenerator + StackingInferenceService + RuleEngine ]
    │
    ▼ in-RAM
[ ALIDataCache : joueurs.parquet + echiquiers.parquet + SHA-256 sigs ]
    │
    ▼ async
[ MongoDB Atlas (audit log only, read-only for /compose flow) ]
```

### 1.2 Assets

| Asset | Sensitivity | Confidentiality | Integrity | Availability |
|-------|-------------|-----------------|-----------|--------------|
| Model weights (MLP champion) | Medium | Low (intent open Phase 6 SDK) | High | Medium |
| Parquet datasets | Low (FFE public) | None (public) | High | Medium |
| FFE Rules JSON (a02.json) | Low | None (public PDF) | High | High |
| Audit log MongoDB | High | Medium (tenant_id Phase 5) | High (non-repudiation) | Medium |
| API keys / JWT secrets | Critical (Phase 5) | High | High | High |
| Predictions / lineage_hash | Medium | Medium (per-tenant Phase 5) | High | Medium |

---

## 2. STRIDE Threats

### 2.1 Spoofing (Identity)

| Threat | Vector | Impact | Mitigation Phase 3 | Mitigation Phase 5 |
|--------|--------|--------|--------------------|--------------------|
| Forged `opponent_club_id` | API caller submits non-existent FFE club ID | Wasted compute, possibly biased aggregate metrics | Pydantic validation on FFE ID format `A12345` | Cross-check vs `joueurs_by_club` keyset → 404 |
| Identity impersonation | Attacker calls `/compose` as another club (multi-tenant Phase 5) | Privacy concern (preferences leaked) | N/A (single-tenant Phase 3) | JWT auth with scope `tenant_id` |
| User-Agent spoofing | Bot disguised as browser to bypass rate limit | DoS | Rate limiting per IP (slowapi) | Bot detection (Vercel BotID Phase 5+ if commercialized) |

### 2.2 Tampering (Integrity)

| Threat | Vector | Impact | Mitigation Phase 3 | Mitigation Phase 5 |
|--------|--------|--------|--------------------|--------------------|
| Malicious `overrides[*]` in `ComposeRequest` | Inject Elo=3000 fake player to skew scenarios | Wrong predictions, captain mislead | Pydantic strict validation : Elo ∈ [500, 3000], FFE ID format. `_row_licence_active` defaults False on suspicious overrides. | Cross-validate against `joueurs.parquet` membership — reject NR_FFE not in club's roster |
| Parquet tampering at rest | Attacker modifies `joueurs.parquet` on disk | All predictions corrupted | SHA-256 stored in `ALIDataCache.parquet_sig_*` at load. Check on next startup vs known good. | DVC versioning (Phase 5 D6/D7) + GPG signed releases |
| FFE Rules JSON tampering | Modify `config/ffe_rules/a02.json` | Wrong rule application | `RuleEngine.lineage_hash()` SHA-256 included in `ScenarioSet.lineage_hash` | Pre-commit `ffe-rules-drift` hook detects unauthorized changes |
| MITM on API call | Network attacker modifies request/response | Wrong predictions in transit | N/A (HTTP localhost dev) | TLS 1.3 mandatory Phase 5 |

### 2.3 Repudiation

| Threat | Vector | Impact | Mitigation Phase 3 | Mitigation Phase 5 |
|--------|--------|--------|--------------------|--------------------|
| Captain denies receiving prediction | "I never got that recommendation" | Dispute resolution impossible | Audit log MongoDB (ISO 27001 A.8.15) per `/compose` call with `lineage_hash`, timestamp UTC, `rule_uuids`, `model_versions` | Tenant-scoped audit + retention policy 7 years (Phase 5 GDPR) |
| Developer denies model version deployed | Production incident attribution | Blame impossible | `model_versions` field in audit metadata + git commit pin | DVC + commit-pinning of model weights (D6/D7) |
| ALI denies generating biased prediction | Class action / FFE complaint | Liability uncovered | Lineage hash reproduces exact inputs given same `lineage_hash` prefix | + Threat Model + Risk Register signed reviews per release |

### 2.4 Information Disclosure

| Threat | Vector | Impact | Mitigation Phase 3 | Mitigation Phase 5 |
|--------|--------|--------|--------------------|--------------------|
| Private player stats leaked | Unauthenticated `/compose` exposes player Elo/club | Low — data is FFE public | Public data only, no GDPR concern | Per-tenant scoping if monetized + clarify what user can see |
| Training data exfiltration via API | Adversary calls `/compose` repeatedly to reverse-engineer training distribution | Low value (data public) | Rate limiting + log analysis | + Differential privacy if needed Phase 7 |
| Audit log leak | Attacker reads MongoDB audit collection | Medium (reveals usage patterns by club) | MongoDB connection string secret in env vars (no commit) | + tenant_id-based scoping + IP allowlist on Atlas |
| Error message leak (stack traces) | 500 response includes internal paths | Info for further attacks | FastAPI handlers return clean messages, log details only server-side | + Sentry sanitization Phase 5 |
| Lineage hash reverse-engineered | Hash includes opponent_club_id + saison + ronde — predictable | Low — already public info | None (lineage_hash is auditable identifier, not secret) | None needed |

### 2.5 Denial of Service (DoS)

| Threat | Vector | Impact | Mitigation Phase 3 | Mitigation Phase 5 |
|--------|--------|--------|--------------------|--------------------|
| Excessive `n_topk` / `n_mc_pairs` in overrides | Caller sets `n_mc_pairs=1000000` | OOM / latency spike | Hard caps : `n_topk ∈ [1, 20]`, `n_mc_pairs ∈ [0, 50]` enforced in `ScenarioGenerator.generate` signature defaults | + Pydantic `Field(le=20)` on schema |
| Cache stampede (cold restart) | Multiple parallel `lifespan` triggers reload | Latency p95 spike | Single in-process cache, mutex (lifespan blocks until ready) | + warm-up health check before traffic routing |
| Slow-loris on POST | Attacker holds open connection sending bytes slowly | Connection pool exhaustion | uvicorn default timeout | + Caddy/Nginx connection timeout 30s |
| Repeated `/compose` from single IP | Burst attack | Compute saturation | slowapi rate limit per IP (existing) | per-`user_club_id` rate limit Phase 5 |
| Pool too small → 50 retries × 20 scenarios | Specially-crafted tiny opponent club causes 1000 RuleEngine validations | CPU spike per request | `_MAX_RETRIES = 50` cap in MC + `max_rounds = 5` in `_merge_and_pad` | + circuit breaker on rejection_rate > 50 % |

### 2.6 Elevation of Privilege

| Threat | Vector | Impact | Mitigation Phase 3 | Mitigation Phase 5 |
|--------|--------|--------|--------------------|--------------------|
| Unauthenticated admin endpoint exposed | If `/admin/*` added later without auth | Full system compromise | N/A (no admin endpoints Phase 3) | JWT auth with role scopes |
| Path traversal in file uploads | Future Phase 6 SDK uploads | Read arbitrary files | N/A (no upload Phase 3) | Path normalization + allowlist |
| Pickle deserialization on model load | `joblib.load` of malicious model file | RCE | Models loaded from trusted local cache only | + signed model artifacts (DVC + GPG) |
| Container escape | If deployed in Docker/k8s Phase 5+ | Full host compromise | N/A | Read-only root filesystem + non-root user |

---

## 3. OWASP API Security Top 10 (2023) Mapping

| OWASP ID | Status Phase 3 | Mitigation |
|----------|---------------|------------|
| API1:2023 Broken Object Level Auth | Deferred Phase 5 | Single-tenant Phase 3 (no per-resource auth needed yet) |
| API2:2023 Broken Auth | Deferred Phase 5 | No auth Phase 3 ; JWT Phase 5 |
| API3:2023 Broken Object Property Level Auth | Deferred Phase 5 | Pydantic schemas don't expose sensitive fields ; full review Phase 5 |
| API4:2023 Unrestricted Resource Consumption | **Mitigated** | Hard caps + rate limit (S2.5) |
| API5:2023 Broken Function Level Auth | Deferred Phase 5 | No admin endpoints Phase 3 |
| API6:2023 Unrestricted Access to Sensitive Business Flows | **Mitigated** | Rate limit + lineage_hash audit |
| API7:2023 Server Side Request Forgery | **N/A** | No outbound URL fetching from user input |
| API8:2023 Security Misconfiguration | Partial | CORS configured `app/main.py`, secrets via env vars only ; full audit Phase 5 |
| API9:2023 Improper Inventory Management | Partial | API versioned `/api/v1/*` ; deprecation policy Phase 5 |
| API10:2023 Unsafe Consumption of APIs | **N/A** | No external API consumed in `/compose` flow |

---

## 4. Pen-test Plan (Phase 5)

Before Phase 5 production deploy :

1. **OWASP ZAP automated scan** on staging environment
2. **Manual fuzzing** of `ComposeRequest.player_overrides` with adversarial payloads
3. **Load test** : `locust` simulating 10 concurrent users × 100 requests
4. **Auth bypass attempts** on JWT scope handling
5. **Pickle / model loading audit** : verify no untrusted deserialization
6. **Secrets audit** : `gitleaks` + manual review of env var handling
7. **MongoDB hardening** : Atlas IP allowlist + role-based access (read-only
   audit collection from app)

---

## 5. Threat Update Cadence

- **Per release** : review threat model, update mitigations, mark new
  threats discovered
- **Per ADR change** : if architecture changes, re-evaluate trust
  boundaries
- **Annually** (Phase 5+) : full pen-test by external party (if commercialized)
- **Per incident** : post-mortem feeds new threat entries

---

## 6. References

- Microsoft STRIDE methodology : Howard, M. & LeBlanc, D. 2003 "Writing
  Secure Code" 2nd ed., Microsoft Press
- ISO/IEC 27001:2022 — Information Security Management Systems
- ISO/IEC 27034:2011 — Application Security
- ISO/IEC 25059:2023 §security characteristic
- OWASP API Security Top 10 (2023) — https://owasp.org/API-Security/

---

## 7. Document Control

| Field | Value |
|-------|-------|
| Document ID | ALICE-THREAT-MODEL-ALI-v1.0.0 |
| Created | 2026-04-28 |
| Status | DRAFT (Phase 3 §G5 fix-on-sight from Model Card audit) |
| Author | Pierre-Alexandre Guillemin |
| LLM Co-author | Claude Opus 4.7 (1M context) |
| Next review | Phase 5 pre-prod (full pen-test) |
| Related | `docs/iso/ALI_MODEL_CARD.md` §11 Ethical, §15 Compliance Cross-Refs |
