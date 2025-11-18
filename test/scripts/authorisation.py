"""CLI tool for EdDSA key pair generation and JWT token creation."""

import json
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Annotated

import typer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

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
def keygen(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save the key files",
        ),
    ] = Path("."),
    private_key_name: Annotated[
        str,
        typer.Option(
            "--private-name",
            "-p",
            help="Filename for private key",
        ),
    ] = DEFAULT_PRIVATE_KEY,
    public_key_name: Annotated[
        str,
        typer.Option(
            "--public-name",
            "-P",
            help="Filename for public key",
        ),
    ] = DEFAULT_PUBLIC_KEY,
) -> None:
    """Generate an EdDSA key pair and save to files."""
    unencrypted_pem_private_key, pem_public_key = generate_EdDSA_key_pair()

    output_dir.mkdir(parents=True, exist_ok=True)

    private_key_path = output_dir / private_key_name
    public_key_path = output_dir / public_key_name

    private_key_path.write_bytes(unencrypted_pem_private_key)
    public_key_path.write_bytes(pem_public_key)

    typer.echo(f"✓ Private key saved to: {private_key_path}")
    typer.echo(f"✓ Public key saved to: {public_key_path}")


@app.command()
def jwt(
    sub: Annotated[
        str,
        typer.Option(
            "--sub",
            "-s",
            help="Subject claim for the JWT",
        ),
    ],
    private_key_path: Annotated[
        Path,
        typer.Option(
            "--private-key",
            "-k",
            help="Path to private key PEM file",
        ),
    ] = Path(DEFAULT_PRIVATE_KEY),
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
) -> None:
    """Create a JWT token using a private key and subject."""
    if not private_key_path.exists():
        typer.echo(f"Error: Private key file not found: {private_key_path}", err=True)
        raise typer.Exit(1)

    private_key_bytes = private_key_path.read_bytes()
    token = generate_json_web_token(
        private_key_bytes=private_key_bytes,
        sub=sub,
        api_root=api_root,
        expiry_hours=expiry_hours,
    )

    typer.echo("\n✓ JWT Token generated:")
    typer.echo(token)


if __name__ == "__main__":
    app()
