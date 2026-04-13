"""
agent/auth/base.py
──────────────────
Abstract base class and shared data types for authentication providers.

To add a new provider (e.g. GitHub, Apple Sign-In):
  1. Subclass ``AuthProvider``
  2. Implement ``get_login_url()`` and ``handle_callback()``
  3. Register in ``agent/auth/__init__.py  →  PROVIDERS``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuthUser:
    """Authenticated user identity returned by any provider.

    Attributes
    ----------
    email : str
        The verified email address (primary key for MongoDB ``users``).
    name : str
        Display name (may be empty if the provider doesn't supply one).
    picture : str
        URL to the user's avatar / profile picture.
    provider : str
        Which provider authenticated this user (``"google"``, etc.).
    raw : dict
        Full provider-specific payload for debugging / future use.
    """

    email: str
    name: str = ""
    picture: str = ""
    provider: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class AuthProvider(ABC):
    """Abstract base for pluggable authentication providers.

    Each concrete provider must expose:
    * ``provider_name``   — short identifier (``"google"``).
    * ``is_configured``   — property: True when the required env vars are set.
    * ``get_login_url()`` — returns the URL the browser should navigate to.
    * ``handle_callback()`` — exchanges the auth code for an ``AuthUser``.
    """

    provider_name: str = ""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if the provider has all required credentials."""
        ...

    @abstractmethod
    def get_login_url(self, state: str = "") -> str:
        """Build the OAuth / SSO redirect URL.

        Parameters
        ----------
        state : str, optional
            Opaque CSRF token echoed back in the callback.
        """
        ...

    @abstractmethod
    def handle_callback(self, code: str) -> AuthUser:
        """Exchange the authorisation code for an ``AuthUser``.

        Parameters
        ----------
        code : str
            The authorisation code received from the provider callback.
        """
        ...
