"""CLI tool for EdDSA key pair generation and JWT token creation."""

import json
import sys
import time
from base64 import urlsafe_b64encode
from typing import Annotated

import typer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from rich import print, print_json

app = typer.Typer()

EXPIRY_AFTER_X_HOURS = 24
DEFAULT_PRIVATE_KEY = "private_key.pem"
DEFAULT_PUBLIC_KEY = "public_key.pem"


def b64encode_nopadding(to_encode: bytes) -> bytes:
    """B64 encode without padding."""
    return urlsafe_b64encode(to_encode).rstrip(b"=")


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


def generate_json_web_token(
    private_key_bytes: bytes,
    sub: str,
    api_root: str = "https://api.example.com",
    expiry_hours: int = EXPIRY_AFTER_X_HOURS,
    email: str | None = None,
) -> str:
    """Generate JWT with private key bytes."""
    private_key = load_pem_private_key(private_key_bytes, password=None)

    header = {
        "typ": "JWT",
        "alg": "EdDSA",
        "crv": "Ed25519",
    }
    payload = {
        "sub": sub,
        "exp": int(time.time() + 60 * 60 * expiry_hours),
        "authorised_hosts": api_root,
    }

    if email is not None:
        payload["email"] = email

    to_sign = (
        b64encode_nopadding(json.dumps(header).encode("utf-8"))
        + b"."
        + b64encode_nopadding(json.dumps(payload).encode("utf-8"))
    )
    signature = b64encode_nopadding(private_key.sign(to_sign))
    token = (to_sign + b"." + signature).decode()
    return token


# CLI commands
@app.command()
def keygen() -> None:
    """Generate an EdDSA key pair and output as JSON.

    Pipe to a .json to save.
    """
    unencrypted_pem_private_key, pem_public_key = generate_EdDSA_key_pair()

    output = {
        "private_key": unencrypted_pem_private_key.decode("utf-8"),
        "public_key": pem_public_key.decode("utf-8"),
    }

    print_json(json.dumps(output))


@app.command()
def jwt(
    sub: Annotated[
        str,
        typer.Option(
            "--sub",
            "-s",
            help="Subject claim for the JWT (should be a UUID string)",
        ),
    ],
    email: Annotated[
        str | None,
        typer.Option(
            "--email",
            "-m",
            help="Optional email address to include in the JWT",
        ),
    ] = None,
    api_root: Annotated[
        str,
        typer.Option(
            "--api-root",
            "-a",
            help="Authorised API host",
        ),
    ] = "https://api.example.com",
    expiry_hours: Annotated[
        int,
        typer.Option(
            "--expiry",
            "-e",
            help="Token expiry in hours",
        ),
    ] = EXPIRY_AFTER_X_HOURS,
    private_key: Annotated[
        typer.FileText | None,
        typer.Option(
            "--key",
            "-k",
            help="Private key file (or use stdin)",
        ),
    ] = None,
) -> None:
    """Generate JWT from private key.
    
    Use with keygen like:

    uv run test/scripts/authorisation.py keygen > keys.json
    
    cat keys.json | \
    jq -r .private_key | \
    uv run test/scripts/authorisation.py jwt \
        --sub a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
        --email user@example.com \
        --api-root http://api.example.com/ \
        --expiry 2
    
    Or with a key file:
    
    uv run test/scripts/authorisation.py jwt \
        --key private_key.pem \
        --sub f9e8d7c6-b5a4-3210-9876-543210fedcba \
        --email user@example.com
    """

    if private_key:
        private_key_pem = private_key.read()
    elif not sys.stdin.isatty():
        private_key_pem = sys.stdin.read()
    else:
        typer.echo(
            "Error: No input piped. Provide a private key via stdin or --key.", err=True
        )
        raise typer.Exit(code=1)

    token = generate_json_web_token(
        private_key_bytes=private_key_pem.encode(),
        sub=sub,
        api_root=api_root,
        expiry_hours=expiry_hours,
        email=email,
    )

    print(token)


if __name__ == "__main__":
    app()
