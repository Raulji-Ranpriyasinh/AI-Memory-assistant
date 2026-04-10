"""
HIPAA compliance utilities — audit logging, consent management,
PII encryption, retention policy, and right-to-erasure.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config.settings import DB_URI, HIPAA_ENCRYPTION_KEY
from app.observability.metrics import metrics


# ── Configuration error ─────────────────────────────────────────────────────

class ConfigurationError(Exception):
    """Raised when required HIPAA configuration is missing."""
    pass


# ── Audit log ────────────────────────────────────────────────────────────────

_AUDIT_LOG_FILE = os.getenv("HIPAA_AUDIT_LOG", "hipaa_audit.jsonl")


def audit_log(
    user_id: str,
    action: str,
    resource: str,
    **kwargs: Any,
) -> None:
    """
    Append JSON record to hipaa_audit.jsonl.
    Fields: ts (UTC ISO), event='hipaa_audit', user_id, action, resource, kwargs.
    Also increments metrics counter.
    """
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": "hipaa_audit",
        "user_id": user_id,
        "action": action,
        "resource": resource,
        **kwargs,
    }

    try:
        with open(_AUDIT_LOG_FILE, "a", buffering=1) as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass

    metrics.log("hipaa_audit", user_id=user_id, action=action, resource=resource)


# ── Consent management ──────────────────────────────────────────────────────

VALID_CONSENT_TYPES = {
    "data_collection",
    "data_sharing",
    "ai_processing",
    "push_notifications",
}


def record_consent(
    user_id: str,
    consent_type: str,
    granted: bool,
    version: str = "1.0",
) -> None:
    """
    Store consent record in PostgresStore under ('consent', user_id).
    consent_type values: 'data_collection', 'data_sharing', 'ai_processing', 'push_notifications'.
    """
    if consent_type not in VALID_CONSENT_TYPES:
        raise ValueError(
            f"Invalid consent type '{consent_type}'. Must be one of: {VALID_CONSENT_TYPES}"
        )

    from langgraph.store.postgres import PostgresStore

    ns = ("consent", user_id)
    consent_data = {
        "consent_type": consent_type,
        "granted": granted,
        "version": version,
        "recorded_at": datetime.utcnow().isoformat() + "Z",
    }

    with PostgresStore.from_conn_string(DB_URI) as store:
        store.put(ns, consent_type, {"data": consent_data})

    audit_log(user_id, "write", "consent", consent_type=consent_type, granted=granted)
    metrics.log("consent_recorded", user_id=user_id, consent_type=consent_type)


def check_consent(user_id: str, consent_type: str) -> bool:
    """
    Return bool. Returns False if no consent record exists (fail-closed).
    """
    from langgraph.store.postgres import PostgresStore

    ns = ("consent", user_id)

    try:
        with PostgresStore.from_conn_string(DB_URI) as store:
            items = store.search(ns)
            for item in items:
                data = item.value.get("data", {})
                if data.get("consent_type") == consent_type:
                    return bool(data.get("granted", False))
    except Exception:
        pass

    # Fail-closed: no consent record → no consent
    metrics.log("consent_missing", user_id=user_id, consent_type=consent_type)
    return False


# ── PII encryption (AES-256-GCM) ────────────────────────────────────────────

def encrypt_pii(data: str) -> str:
    """
    AES-256-GCM encryption using HIPAA_ENCRYPTION_KEY.
    Return base64-encoded ciphertext with IV prepended.
    Raise ConfigurationError if key not set.
    """
    import base64

    if not HIPAA_ENCRYPTION_KEY:
        raise ConfigurationError(
            "HIPAA_ENCRYPTION_KEY not set. Cannot encrypt PII."
        )

    key = HIPAA_ENCRYPTION_KEY.encode("utf-8")
    # Ensure key is 32 bytes (AES-256)
    if len(key) < 32:
        key = key.ljust(32, b"\0")
    elif len(key) > 32:
        key = key[:32]

    aesgcm = AESGCM(key)
    iv = os.urandom(12)  # 96-bit IV for GCM
    ciphertext = aesgcm.encrypt(iv, data.encode("utf-8"), None)

    # Prepend IV to ciphertext, then base64-encode
    return base64.b64encode(iv + ciphertext).decode("utf-8")


def decrypt_pii(encrypted: str) -> str:
    """
    Reverse of encrypt_pii. Extract IV from prepended bytes. Decrypt and return plaintext.
    """
    import base64

    if not HIPAA_ENCRYPTION_KEY:
        raise ConfigurationError(
            "HIPAA_ENCRYPTION_KEY not set. Cannot decrypt PII."
        )

    key = HIPAA_ENCRYPTION_KEY.encode("utf-8")
    if len(key) < 32:
        key = key.ljust(32, b"\0")
    elif len(key) > 32:
        key = key[:32]

    aesgcm = AESGCM(key)
    raw = base64.b64decode(encrypted)

    # Extract IV (first 12 bytes) and ciphertext
    iv = raw[:12]
    ciphertext = raw[12:]

    plaintext = aesgcm.decrypt(iv, ciphertext, None)
    return plaintext.decode("utf-8")


# ── Retention policy ────────────────────────────────────────────────────────

# Retention periods by category (in days). None = never delete.
RETENTION_PERIODS: Dict[str, Optional[int]] = {
    "cgm_pattern": 365,
    "mood_pattern": 180,
    "dietary": 365,
    "activity": 365,
    "medication": None,      # never delete
    "identity": None,         # never delete
    "health_condition": None, # never delete
}


def apply_retention_policy(user_id: str) -> int:
    """
    Retention periods by category.
    Delete LTM memories older than their category limit.
    Return count deleted.
    """
    from app.memory.pinecone_ltm import PineconeLTMManager

    ltm = PineconeLTMManager()
    all_memories = ltm.list_all_memories(user_id)

    deleted = 0
    now = datetime.utcnow()

    for memory in all_memories:
        category = memory.get("category", "")
        created_at_str = memory.get("created_at") or memory.get("ingested_at")
        if not created_at_str:
            continue

        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            # Make naive for comparison
            created_at = created_at.replace(tzinfo=None)
        except (ValueError, AttributeError):
            continue

        retention_days = RETENTION_PERIODS.get(category)
        if retention_days is None:
            # Never delete
            continue

        age_days = (now - created_at).days
        if age_days > retention_days:
            memory_id = memory.get("id") or memory.get("pinecone_id")
            if memory_id:
                try:
                    ltm._index.delete(ids=[memory_id])
                    deleted += 1
                    audit_log(
                        user_id, "delete", "ltm_memory",
                        memory_id=memory_id, category=category, reason="retention_policy",
                    )
                except Exception:
                    continue

    metrics.log("retention_applied", user_id=user_id, deleted=deleted)
    return deleted


# ── Right to erasure (GDPR / HIPAA) ────────────────────────────────────────

def delete_all_user_data(user_id: str) -> Dict[str, Any]:
    """
    Right to erasure (GDPR/HIPAA).
    Delete: all Pinecone LTM vectors for user, all PostgresStore namespaces
    (candidates, summaries, nudges, programs, consent),
    redact user_id from audit log (replace with [REDACTED]).
    Return {pinecone_deleted, postgres_deleted, audit_redacted}.
    """
    from langgraph.store.postgres import PostgresStore

    pinecone_deleted = 0
    postgres_deleted = 0
    audit_redacted = 0

    # 1. Delete all Pinecone LTM vectors for user
    try:
        from app.memory.pinecone_ltm import PineconeLTMManager
        ltm = PineconeLTMManager()
        all_memories = ltm.list_all_memories(user_id)
        ids_to_delete = [
            m.get("id") or m.get("pinecone_id")
            for m in all_memories
            if m.get("id") or m.get("pinecone_id")
        ]
        if ids_to_delete:
            ltm._index.delete(ids=ids_to_delete)
            pinecone_deleted = len(ids_to_delete)
    except Exception as exc:
        metrics.log("erasure_pinecone_failed", user_id=user_id, error=str(exc))

    # 2. Delete all PostgresStore namespaces
    namespaces = ["candidates", "summaries", "nudges", "programs", "consent"]
    try:
        with PostgresStore.from_conn_string(DB_URI) as store:
            for ns_name in namespaces:
                ns = (ns_name, user_id)
                items = store.search(ns)
                for item in items:
                    try:
                        store.delete(ns, item.key)
                        postgres_deleted += 1
                    except Exception:
                        continue
    except Exception as exc:
        metrics.log("erasure_postgres_failed", user_id=user_id, error=str(exc))

    # 3. Redact user_id from audit log
    try:
        if os.path.exists(_AUDIT_LOG_FILE):
            redacted_lines = []
            with open(_AUDIT_LOG_FILE, "r") as f:
                for line in f:
                    if user_id in line:
                        redacted_lines.append(line.replace(user_id, "[REDACTED]"))
                        audit_redacted += 1
                    else:
                        redacted_lines.append(line)

            with open(_AUDIT_LOG_FILE, "w") as f:
                f.writelines(redacted_lines)
    except OSError:
        pass

    audit_log(
        user_id, "delete", "all_user_data",
        pinecone_deleted=pinecone_deleted,
        postgres_deleted=postgres_deleted,
        audit_redacted=audit_redacted,
    )
    metrics.log(
        "erasure_complete",
        user_id="[REDACTED]",
        pinecone_deleted=pinecone_deleted,
        postgres_deleted=postgres_deleted,
        audit_redacted=audit_redacted,
    )

    return {
        "pinecone_deleted": pinecone_deleted,
        "postgres_deleted": postgres_deleted,
        "audit_redacted": audit_redacted,
    }
