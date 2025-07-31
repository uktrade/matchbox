"""Utilities for JWT API authorisation.

NOTE: A secure set-up requires the JWT generation logic to not live on the client.
Instead, client JWT should be generated within a secure environment with access to
the private key.

This file supports the edge case of a trusted client bypassing the JWT mechanism
altogether, which is not recommended in general.

"""

import json
import time
from base64 import urlsafe_b64encode

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from matchbox.client._settings import settings

EXPIRY_AFTER_X_HOURS = 24


def b64encode_nopadding(to_encode):
    """B64 encode."""
    return urlsafe_b64encode(to_encode).rstrip(b"=")


def generate_json_web_token(sub):
    """Generate JWT with private API Key."""
    private_key = load_pem_private_key(
        settings.private_key.get_secret_value().encode(), password=None
    )
    header = {
        "typ": "JWT",
        "alg": "EdDSA",
        "crv": "Ed25519",
    }
    payload = {
        "sub": sub,
        "exp": int(time.time() + 60 * 60 * EXPIRY_AFTER_X_HOURS),
        "authorised_hosts": settings.api_root,
    }
    to_sign = (
        b64encode_nopadding(json.dumps(header).encode("utf-8"))
        + b"."
        + b64encode_nopadding(json.dumps(payload).encode("utf-8"))
    )
    signature = b64encode_nopadding(private_key.sign(to_sign))
    token = (to_sign + b"." + signature).decode()
    return token


def generate_EdDSA_key_pair() -> tuple[bytes, bytes]:
    """Generate private and public key pair."""
    private_key = Ed25519PrivateKey.generate()

    unencrypted_pem_private_key = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    pem_public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return unencrypted_pem_private_key, pem_public_key
