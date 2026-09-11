"""Signed assistant turns: the Python side of lib/assistant-signature.ts.

Why they exist is written there. In short: the conversation history is client-supplied, so the
Next.js route drops any assistant turn whose text does not carry a valid HMAC from this system.
Once Python generates the answers (step 2.4), Python has to sign them, with the same scheme and
the same ASSISTANT_SIGNING_SECRET, or every follow-up would lose the turn before it. Verifying
stays in TypeScript: the route parses the request before anything reaches this service.

Pinned two ways: tests/test_signing.py checks known vectors computed with Node's crypto, and the
Vitest contract test (tests/python-stream-contract.test.ts) verifies the signatures in the
golden Python streams with the real verifyAssistantText.
"""

import base64
import hashlib
import hmac

# Prefixed into the signed payload and onto the signature (VERSION in lib/assistant-signature.ts;
# tests/test_ts_parity.py pins it).
VERSION = "v1"


def sign_assistant_text(text: str, secret: str) -> str:
    """Sign one assistant answer, exactly as it was streamed to the client.

    HMAC-SHA256 over "v1:" + text, both as UTF-8, written as base64url without padding (what
    Node's digest("base64url") produces), prefixed with "v1.".
    """
    mac = hmac.new(secret.encode(), f"{VERSION}:{text}".encode(), hashlib.sha256).digest()
    return f"{VERSION}." + base64.urlsafe_b64encode(mac).rstrip(b"=").decode("ascii")
