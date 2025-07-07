"""Utilities for JWT API authorisation."""

import time

import jwt

EXPIRY_AFTER_X_HOURS = 24


def generate_json_web_token(
    sub: str, private_key: str, algorithm: str = "HS256"
) -> str:
    """Generate JWT with private API Key."""
    # Type=None due to header encoding mismatch with earlier pyJWT library versions
    header = {
        "alg": algorithm,
        "typ": None,
    }
    payload = {
        "sub": sub,
        "exp": int(time.time() + 60 * 60 * EXPIRY_AFTER_X_HOURS),
    }
    return jwt.encode(
        payload=payload, key=private_key, algorithm=algorithm, headers=header
    )
