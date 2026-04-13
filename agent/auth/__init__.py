"""
agent/auth — Pluggable authentication providers for FITGEN.AI.

Currently ships with Google OAuth 2.0; add new providers by subclassing
``AuthProvider`` and registering them in ``PROVIDERS``.
"""

from agent.auth.base import AuthProvider, AuthUser
from agent.auth.google_auth import GoogleAuthProvider

# Registry: provider_name → class
PROVIDERS: dict[str, type[AuthProvider]] = {
    "google": GoogleAuthProvider,
}

__all__ = [
    "AuthProvider",
    "AuthUser",
    "GoogleAuthProvider",
    "PROVIDERS",
]
