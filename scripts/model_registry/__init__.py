"""Model Registry pour ALICE - ISO 42001/5259/27001.

Ce package centralise la gestion des modèles ML:
- Sauvegarde/chargement avec validation d'intégrité
- Traçabilité des données (data lineage)
- Sécurité (signatures HMAC, chiffrement AES-256)
- Drift monitoring
- Rollback de versions

Structure:
- dataclasses.py: Structures de données (DataLineage, ModelArtifact, etc.)
- utils.py: Utilitaires (checksums, git info, etc.)
- security.py: Cryptographie (HMAC-SHA256, AES-256-GCM)
- drift.py: Détection de drift (PSI, métriques)
- validation.py: Validation schema DataFrame
- artifacts.py: Sauvegarde/chargement modèles
- versioning.py: Rollback et API haut niveau

Conformité ISO/IEC 42001, 5259, 27001, 5055, 29119.
"""

# Dataclasses
# Artifacts
from scripts.model_registry.artifacts import (
    MODEL_FORMATS,
    ONNX_OPSET_VERSION,
    export_to_onnx,
    extract_feature_importance,
    load_model_with_validation,
    save_model_artifact,
    validate_model_integrity,
)
from scripts.model_registry.dataclasses import (
    DataLineage,
    EnvironmentInfo,
    ModelArtifact,
    ProductionModelCard,
)

# Drift Monitoring
from scripts.model_registry.drift import (
    ACCURACY_DROP_THRESHOLD,
    ELO_SHIFT_THRESHOLD,
    PSI_THRESHOLD_CRITICAL,
    PSI_THRESHOLD_WARNING,
    DriftMetrics,
    DriftReport,
    add_round_to_drift_report,
    check_drift_status,
    compute_drift_metrics,
    compute_psi,
    create_drift_report,
    load_drift_report,
    save_drift_report,
)

# Drift Monitor (ISO 23894) - Main
from scripts.model_registry.drift_monitor import (
    analyze_feature_drift,
    monitor_drift,
)

# Drift Monitor (ISO 23894) - Stats
from scripts.model_registry.drift_stats import (
    compute_chi2_test,
    compute_js_divergence,
    compute_ks_test,
)

# Drift Monitor (ISO 23894) - Types
from scripts.model_registry.drift_types import (
    KS_PVALUE_CRITICAL,
    KS_PVALUE_OK,
    KS_PVALUE_WARNING,
    PSI_THRESHOLD_OK,
    DriftMonitorResult,
    DriftSeverity,
    DriftType,
    FeatureDriftResult,
)

# Input Validator (ISO 24029) - Types
from scripts.model_registry.input_types import (
    DEFAULT_STD_TOLERANCE,
    OOD_REJECTION_THRESHOLD,
    FeatureBounds,
    FeatureValidationResult,
    InputBoundsConfig,
    InputValidationResult,
    OODAction,
    OODSeverity,
)

# Input Validator (ISO 24029) - Main
from scripts.model_registry.input_validator import (
    compute_feature_bounds,
    create_bounds_config,
    load_bounds_config,
    save_bounds_config,
    validate_batch,
    validate_input,
)

# Security
from scripts.model_registry.security import (
    ENCRYPTED_EXTENSION,
    ENV_ENCRYPTION_KEY,
    ENV_SIGNING_KEY,
    compute_model_signature,
    decrypt_model_directory,
    decrypt_model_file,
    encrypt_model_directory,
    encrypt_model_file,
    generate_encryption_key,
    generate_signing_key,
    get_key_from_env,
    get_signing_key_from_env,
    load_encryption_key,
    load_signing_key,
    save_encryption_key,
    save_signing_key,
    verify_model_signature,
)

# Utilities
from scripts.model_registry.utils import (
    compute_data_lineage,
    compute_dataframe_hash,
    compute_file_checksum,
    get_environment_info,
    get_git_info,
    get_package_versions,
)

# Validation
from scripts.model_registry.validation import (
    DEFAULT_MAX_VERSIONS,
    ELO_MAX,
    ELO_MIN,
    ELO_WARNING_HIGH,
    ELO_WARNING_LOW,
    REQUIRED_NUMERIC_COLUMNS,
    REQUIRED_TRAIN_COLUMNS,
    SchemaValidationResult,
    apply_retention_policy,
    get_retention_status,
    list_model_versions,
    validate_dataframe_schema,
    validate_train_valid_test_schema,
)

# Versioning & High-Level API
from scripts.model_registry.versioning import (
    create_production_model_card,
    get_current_version,
    rollback_to_version,
    save_production_model_card,
    save_production_models,
)

__all__ = [
    # Dataclasses
    "DataLineage",
    "EnvironmentInfo",
    "ModelArtifact",
    "ProductionModelCard",
    "DriftMetrics",
    "DriftReport",
    "SchemaValidationResult",
    # Constants
    "MODEL_FORMATS",
    "ONNX_OPSET_VERSION",
    "ENCRYPTED_EXTENSION",
    "ENV_ENCRYPTION_KEY",
    "ENV_SIGNING_KEY",
    "PSI_THRESHOLD_WARNING",
    "PSI_THRESHOLD_CRITICAL",
    "ACCURACY_DROP_THRESHOLD",
    "ELO_SHIFT_THRESHOLD",
    "DEFAULT_MAX_VERSIONS",
    "ELO_MIN",
    "ELO_MAX",
    "ELO_WARNING_LOW",
    "ELO_WARNING_HIGH",
    "REQUIRED_TRAIN_COLUMNS",
    "REQUIRED_NUMERIC_COLUMNS",
    # Utils
    "compute_file_checksum",
    "compute_dataframe_hash",
    "get_git_info",
    "get_package_versions",
    "get_environment_info",
    "compute_data_lineage",
    # Security
    "generate_signing_key",
    "compute_model_signature",
    "verify_model_signature",
    "save_signing_key",
    "load_signing_key",
    "get_signing_key_from_env",
    "get_key_from_env",
    "generate_encryption_key",
    "save_encryption_key",
    "load_encryption_key",
    "encrypt_model_file",
    "decrypt_model_file",
    "encrypt_model_directory",
    "decrypt_model_directory",
    # Drift (legacy)
    "compute_psi",
    "compute_drift_metrics",
    "create_drift_report",
    "add_round_to_drift_report",
    "save_drift_report",
    "load_drift_report",
    "check_drift_status",
    # Drift Monitor (ISO 23894)
    "DriftSeverity",
    "DriftType",
    "FeatureDriftResult",
    "DriftMonitorResult",
    "compute_ks_test",
    "compute_chi2_test",
    "compute_js_divergence",
    "analyze_feature_drift",
    "monitor_drift",
    "KS_PVALUE_OK",
    "KS_PVALUE_WARNING",
    "KS_PVALUE_CRITICAL",
    # Validation
    "validate_dataframe_schema",
    "validate_train_valid_test_schema",
    "apply_retention_policy",
    "get_retention_status",
    "list_model_versions",
    # Input Validator (ISO 24029)
    "OODSeverity",
    "OODAction",
    "FeatureBounds",
    "FeatureValidationResult",
    "InputValidationResult",
    "InputBoundsConfig",
    "compute_feature_bounds",
    "create_bounds_config",
    "validate_input",
    "validate_batch",
    "save_bounds_config",
    "load_bounds_config",
    "DEFAULT_STD_TOLERANCE",
    "OOD_REJECTION_THRESHOLD",
    "PSI_THRESHOLD_OK",
    # Artifacts
    "extract_feature_importance",
    "export_to_onnx",
    "save_model_artifact",
    "validate_model_integrity",
    "load_model_with_validation",
    # Versioning & High-Level API
    "rollback_to_version",
    "get_current_version",
    "create_production_model_card",
    "save_production_model_card",
    "save_production_models",
]
