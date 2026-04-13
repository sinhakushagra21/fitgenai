"""
agent/auth/google_auth.py
─────────────────────────
Google OAuth 2.0 authentication provider.

Uses Google's standard OAuth2 endpoints directly via ``requests`` — no
dependency on ``google-auth-oauthlib`` (that library is only used by the
Calendar integration which needs different scopes).

Required ``.env`` variables::

    GOOGLE_CLIENT_ID=<your-web-client-id>.apps.googleusercontent.com
    GOOGLE_CLIENT_SECRET=<your-web-client-secret>

Optional::

    GOOGLE_REDIRECT_URI=http://localhost:8501   # default for local dev
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any
from urllib.parse import urlencode

import requests

from agent.auth.base import AuthProvider, AuthUser

logger = logging.getLogger("fitgen.auth.google")

# ── Google OAuth 2.0 endpoints ───────────────────────────────────
_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
_LOGIN_SCOPES = "openid email profile"


class GoogleAuthProvider(AuthProvider):
    """Google OAuth 2.0 provider for email-based login.

    The same ``GOOGLE_CLIENT_ID`` / ``GOOGLE_CLIENT_SECRET`` pair used by
    Calendar integration works here — only the requested *scopes* differ
    (``openid email profile`` for login vs. calendar scopes for sync).
    """

    provider_name = "google"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ) -> None:
        self.client_id = (
            client_id if client_id is not None
            else os.getenv("GOOGLE_CLIENT_ID", "")
        )
        self.client_secret = (
            client_secret if client_secret is not None
            else os.getenv("GOOGLE_CLIENT_SECRET", "")
        )
        self.redirect_uri = (
            redirect_uri if redirect_uri is not None
            else os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")
        )

    # ── AuthProvider interface ───────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """True when both client ID and secret are present."""
        return bool(self.client_id and self.client_secret)

    def get_login_url(self, state: str = "") -> str:
        """Generate the Google consent URL for login.

        The ``state`` parameter is prefixed with ``login_`` so the
        Streamlit callback handler can distinguish login vs. calendar
        OAuth redirects.
        """
        if not self.is_configured:
            raise ValueError(
                "Google OAuth not configured. "
                "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env."
            )

        if not state:
            state = f"login_{secrets.token_urlsafe(16)}"

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": _LOGIN_SCOPES,
            "access_type": "offline",
            "prompt": "select_account",
            "state": state,
        }
        url = f"{_AUTH_ENDPOINT}?{urlencode(params)}"
        logger.info("Generated Google login URL (state=%s…)", state[:20])
        return url

    def handle_callback(self, code: str) -> AuthUser:
        """Exchange authorisation code for user identity.

        Steps:
          1. POST code to Google's token endpoint → access_token
          2. GET Google's userinfo endpoint → email, name, picture
          3. Return as ``AuthUser``
        """
        # Step 1: exchange code for tokens
        token_resp = requests.post(
            _TOKEN_ENDPOINT,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=15,
        )
        token_resp.raise_for_status()
        tokens: dict[str, Any] = token_resp.json()
        logger.debug("Token exchange successful")

        # Step 2: get user info
        info_resp = requests.get(
            _USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=10,
        )
        info_resp.raise_for_status()
        user_info: dict[str, Any] = info_resp.json()

        email = user_info.get("email", "")
        name = user_info.get("name", "")
        picture = user_info.get("picture", "")

        if not email:
            raise ValueError("Google login succeeded but no email was returned.")

        logger.info("Google login successful: email=%s name=%s", email, name)

        return AuthUser(
            email=email,
            name=name,
            picture=picture,
            provider="google",
            raw=user_info,
        )
