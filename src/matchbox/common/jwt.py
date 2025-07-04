"""Utilities for JWT API authorisation."""

import time

import jwt


def generate_json_web_token(
    sub: str, private_key: str, algorithm: str = "HS256", exp: int | None = None
) -> str:
    """Generate JWT with private API Key."""
    header = {
        "typ": "JWT",
        "alg": algorithm,
    }
    payload = {
        "sub": sub,
    }
    if exp:
        payload["exp"] = exp
    else:
        payload["exp"] = int(time.time() + 60 * 60 * 24)

    return jwt.encode(
        payload=payload, key=private_key, algorithm=algorithm, headers=header
    )
