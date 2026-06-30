"""Route HTTPS verification through the operating system trust store.

Corporate machines commonly sit behind a TLS-intercepting proxy whose root CA
is installed in the system keychain but not in certifi's bundle. huggingface_hub
(and requests) verify against certifi by default, so model downloads fail with
``CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain``.
truststore makes the stdlib ``ssl`` module consult the OS trust store instead,
which already trusts that proxy CA.
"""

from __future__ import annotations

import logging
import os
import ssl

logger = logging.getLogger(__name__)

DISABLE_ENV_VAR = "CHIRP_DISABLE_TRUSTSTORE"


def enable_system_trust_store() -> None:
    """Inject truststore into ``ssl`` so HTTPS uses the OS trust store.

    Idempotent, best-effort, and opt-out via ``CHIRP_DISABLE_TRUSTSTORE``. Any
    failure leaves the default certifi-backed verification in place rather than
    breaking startup.
    """
    if os.environ.get(DISABLE_ENV_VAR):
        return
    try:
        import truststore
    except ImportError:
        return
    if ssl.SSLContext is truststore.SSLContext:
        return
    try:
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - trust-store setup must never break startup
        logger.debug(
            "truststore injection failed; using default certifi bundle",
            exc_info=True,
        )
