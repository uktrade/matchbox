"""Utilities for JWT API authorisation."""

import jwt


def generate_json_web_token(
    sub: str, private_key: str, algorithm: str = "HS256"
) -> str:
    """Generate JWT with private API Key."""
    header = {
        "typ": "JWT",
        "alg": algorithm,
    }
    payload = {
        "sub": sub,
    }
    return jwt.encode(
        payload=payload, key=private_key, algorithm=algorithm, headers=header
    )
