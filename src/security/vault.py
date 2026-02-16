"""
Secure Vault - Encrypted Secrets Manager
==========================================
Encrypts API keys using Fernet (AES-128-CBC).
Keys are decrypted only in memory, never stored in plain text.
"""

import os
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecureVault:
    """
    Manages encrypted storage and retrieval of sensitive API credentials.

    Security Features:
    - AES-128 encryption via Fernet
    - Master key derived from environment variable
    - Keys decrypted only in memory
    - No plain text secrets on disk
    """

    def __init__(self):
        """Initialize vault with encryption key from environment."""
        self._fernet = self._initialize_encryption()
        self._cached_keys = None

    def _initialize_encryption(self) -> Fernet:
        """
        Initialize Fernet encryption using master key from environment.

        The ENCRYPTION_KEY env var should be a Fernet key (base64 encoded).
        If not set, generates one (for first-time setup only).
        """
        master_key = os.environ.get("ENCRYPTION_KEY")

        if not master_key:
            # First-time setup: generate a key
            # In production, this should be set in Railway dashboard
            logger.warning("ENCRYPTION_KEY not set! Generating new key...")
            logger.warning("Set this in Railway dashboard for production!")
            master_key = Fernet.generate_key().decode()
            logger.info(f"Generated ENCRYPTION_KEY: {master_key}")
            logger.info("Save this key securely and set it as environment variable!")

        try:
            return Fernet(master_key.encode() if isinstance(master_key, str) else master_key)
        except Exception as e:
            logger.error(f"Invalid ENCRYPTION_KEY format: {e}")
            raise ValueError("ENCRYPTION_KEY must be a valid Fernet key")

    def encrypt_secret(self, plain_text: str) -> str:
        """
        Encrypt a secret string.

        Args:
            plain_text: The secret to encrypt (e.g., API key)

        Returns:
            Base64-encoded encrypted string
        """
        if not plain_text:
            raise ValueError("Cannot encrypt empty secret")

        encrypted = self._fernet.encrypt(plain_text.encode())
        return encrypted.decode()

    def decrypt_secret(self, encrypted_text: str) -> str:
        """
        Decrypt an encrypted secret.

        Args:
            encrypted_text: Base64-encoded encrypted string

        Returns:
            Decrypted plain text
        """
        if not encrypted_text:
            raise ValueError("Cannot decrypt empty string")

        try:
            decrypted = self._fernet.decrypt(encrypted_text.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise ValueError("Decryption failed - invalid key or corrupted data")

    def load_api_keys(self) -> dict:
        """
        Load and decrypt all OKX API credentials from environment.

        Expected environment variables:
        - OKX_API_KEY_ENCRYPTED: Encrypted API key
        - OKX_SECRET_ENCRYPTED: Encrypted secret key
        - OKX_PASSPHRASE_ENCRYPTED: Encrypted passphrase

        For UNENCRYPTED keys (development only):
        - OKX_API_KEY
        - OKX_SECRET_KEY
        - OKX_PASSPHRASE

        Returns:
            dict with 'api_key', 'secret_key', 'passphrase'
        """
        if self._cached_keys:
            return self._cached_keys

        # Try encrypted keys first (production)
        encrypted_api_key = os.environ.get("OKX_API_KEY_ENCRYPTED")
        encrypted_secret = os.environ.get("OKX_SECRET_ENCRYPTED")
        encrypted_passphrase = os.environ.get("OKX_PASSPHRASE_ENCRYPTED")

        if encrypted_api_key and encrypted_secret and encrypted_passphrase:
            logger.info("Loading encrypted API keys...")
            self._cached_keys = {
                "api_key": self.decrypt_secret(encrypted_api_key),
                "secret_key": self.decrypt_secret(encrypted_secret),
                "passphrase": self.decrypt_secret(encrypted_passphrase)
            }
            return self._cached_keys

        # Fallback to unencrypted (development only)
        api_key = os.environ.get("OKX_API_KEY")
        secret_key = os.environ.get("OKX_SECRET_KEY")
        passphrase = os.environ.get("OKX_PASSPHRASE")

        if api_key and secret_key and passphrase:
            logger.warning("Using UNENCRYPTED API keys - OK for development only!")
            self._cached_keys = {
                "api_key": api_key,
                "secret_key": secret_key,
                "passphrase": passphrase
            }
            return self._cached_keys

        raise ValueError(
            "API keys not found! Set either:\n"
            "  - OKX_API_KEY_ENCRYPTED, OKX_SECRET_ENCRYPTED, OKX_PASSPHRASE_ENCRYPTED (production)\n"
            "  - OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE (development)"
        )

    def clear_cache(self):
        """Clear cached keys from memory (for security)."""
        self._cached_keys = None
        logger.info("Cleared cached API keys from memory")


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key.

    Use this to create the ENCRYPTION_KEY for Railway dashboard.
    Run once: python -c "from src.security.vault import generate_encryption_key; print(generate_encryption_key())"
    """
    return Fernet.generate_key().decode()


def encrypt_keys_helper():
    """
    Helper to encrypt API keys for first-time setup.

    Usage:
        1. Set ENCRYPTION_KEY environment variable
        2. Run this function with plain text keys
        3. Copy encrypted values to Railway dashboard
    """
    import getpass

    print("\n=== API Key Encryption Helper ===\n")
    print("This will encrypt your OKX API keys for secure storage.")
    print("Make sure ENCRYPTION_KEY is set in your environment.\n")

    vault = SecureVault()

    api_key = getpass.getpass("Enter OKX API Key: ")
    secret_key = getpass.getpass("Enter OKX Secret Key: ")
    passphrase = getpass.getpass("Enter OKX Passphrase: ")

    print("\n=== Encrypted Values (copy to Railway dashboard) ===\n")
    print(f"OKX_API_KEY_ENCRYPTED={vault.encrypt_secret(api_key)}")
    print(f"OKX_SECRET_ENCRYPTED={vault.encrypt_secret(secret_key)}")
    print(f"OKX_PASSPHRASE_ENCRYPTED={vault.encrypt_secret(passphrase)}")
    print("\n=== Done! ===\n")


if __name__ == "__main__":
    # Run encryption helper when executed directly
    encrypt_keys_helper()
