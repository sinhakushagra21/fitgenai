"""
tests/test_auth.py
──────────────────
Tests for the pluggable authentication module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.auth.base import AuthProvider, AuthUser
from agent.auth.google_auth import GoogleAuthProvider


# ── AuthUser ─────────────────────────────────────────────────────


class TestAuthUser:
    """AuthUser dataclass tests."""

    def test_create_minimal(self):
        user = AuthUser(email="test@example.com")
        assert user.email == "test@example.com"
        assert user.name == ""
        assert user.picture == ""
        assert user.provider == ""
        assert user.raw == {}

    def test_create_full(self):
        user = AuthUser(
            email="kushagra@fitgen.ai",
            name="Kushagra",
            picture="https://example.com/pic.jpg",
            provider="google",
            raw={"id": "123"},
        )
        assert user.email == "kushagra@fitgen.ai"
        assert user.name == "Kushagra"
        assert user.picture == "https://example.com/pic.jpg"
        assert user.provider == "google"
        assert user.raw == {"id": "123"}

    def test_immutable(self):
        user = AuthUser(email="test@example.com")
        with pytest.raises(AttributeError):
            user.email = "other@example.com"  # type: ignore


# ── GoogleAuthProvider ───────────────────────────────────────────


class TestGoogleAuthProvider:
    """GoogleAuthProvider tests."""

    def test_not_configured_when_empty(self):
        provider = GoogleAuthProvider(client_id="", client_secret="")
        assert not provider.is_configured

    def test_configured_when_set(self):
        provider = GoogleAuthProvider(client_id="cid", client_secret="csec")
        assert provider.is_configured

    def test_provider_name(self):
        provider = GoogleAuthProvider(client_id="cid", client_secret="csec")
        assert provider.provider_name == "google"

    def test_get_login_url_raises_when_not_configured(self):
        provider = GoogleAuthProvider(client_id="", client_secret="")
        with pytest.raises(ValueError, match="not configured"):
            provider.get_login_url()

    def test_get_login_url_contains_required_params(self):
        provider = GoogleAuthProvider(
            client_id="test-client-id",
            client_secret="test-secret",
            redirect_uri="http://localhost:8501",
        )
        url = provider.get_login_url(state="login_test123")
        assert "accounts.google.com" in url
        assert "client_id=test-client-id" in url
        assert "redirect_uri=http" in url
        assert "scope=openid+email+profile" in url
        assert "state=login_test123" in url
        assert "response_type=code" in url

    def test_get_login_url_auto_generates_state(self):
        provider = GoogleAuthProvider(
            client_id="cid", client_secret="csec",
        )
        url = provider.get_login_url()
        assert "state=login_" in url

    @patch("agent.auth.google_auth.requests")
    def test_handle_callback_success(self, mock_requests):
        """Simulate a successful Google OAuth callback."""
        # Mock token exchange
        mock_token_resp = MagicMock()
        mock_token_resp.json.return_value = {
            "access_token": "ya29.test-token",
            "token_type": "Bearer",
        }
        mock_token_resp.raise_for_status = MagicMock()

        # Mock userinfo response
        mock_info_resp = MagicMock()
        mock_info_resp.json.return_value = {
            "email": "kushagra@gmail.com",
            "name": "Kushagra Sinha",
            "picture": "https://lh3.googleusercontent.com/photo.jpg",
        }
        mock_info_resp.raise_for_status = MagicMock()

        mock_requests.post.return_value = mock_token_resp
        mock_requests.get.return_value = mock_info_resp

        provider = GoogleAuthProvider(
            client_id="cid", client_secret="csec",
            redirect_uri="http://localhost:8501",
        )
        user = provider.handle_callback("test-auth-code")

        assert user.email == "kushagra@gmail.com"
        assert user.name == "Kushagra Sinha"
        assert user.picture == "https://lh3.googleusercontent.com/photo.jpg"
        assert user.provider == "google"
        assert user.raw["email"] == "kushagra@gmail.com"

        # Verify the token exchange was called correctly
        mock_requests.post.assert_called_once()
        call_kwargs = mock_requests.post.call_args
        assert call_kwargs[1]["data"]["code"] == "test-auth-code"
        assert call_kwargs[1]["data"]["grant_type"] == "authorization_code"

    @patch("agent.auth.google_auth.requests")
    def test_handle_callback_no_email_raises(self, mock_requests):
        """Raise ValueError if Google returns no email."""
        mock_token_resp = MagicMock()
        mock_token_resp.json.return_value = {"access_token": "tok"}
        mock_token_resp.raise_for_status = MagicMock()

        mock_info_resp = MagicMock()
        mock_info_resp.json.return_value = {"name": "No Email User"}
        mock_info_resp.raise_for_status = MagicMock()

        mock_requests.post.return_value = mock_token_resp
        mock_requests.get.return_value = mock_info_resp

        provider = GoogleAuthProvider(client_id="cid", client_secret="csec")
        with pytest.raises(ValueError, match="no email"):
            provider.handle_callback("code")


# ── AuthProvider ABC ─────────────────────────────────────────────


class TestAuthProviderABC:
    """Verify the abstract base class contract."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AuthProvider()  # type: ignore

    def test_subclass_must_implement_all(self):
        class Incomplete(AuthProvider):
            provider_name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore

    def test_concrete_subclass_works(self):
        class FakeProvider(AuthProvider):
            provider_name = "fake"

            @property
            def is_configured(self):
                return True

            def get_login_url(self, state=""):
                return "https://fake.provider/auth"

            def handle_callback(self, code):
                return AuthUser(email="fake@test.com", provider="fake")

        p = FakeProvider()
        assert p.is_configured
        assert "fake.provider" in p.get_login_url()
        user = p.handle_callback("code123")
        assert user.email == "fake@test.com"
