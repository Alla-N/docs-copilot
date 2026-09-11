"""signing.py against Node's crypto: the vectors below were computed with the exact expression
lib/assistant-signature.ts uses (createHmac("sha256", secret).update(`v1:${text}`)
.digest("base64url")) on Node 24. The Vitest contract test checks the other direction: the real
verifyAssistantText accepts the signatures in the golden Python streams.
"""

import pytest

from copilot_agent.signing import sign_assistant_text

# Non-ASCII on purpose: e acute, an em dash, a check mark, U+2028 and an astral emoji (two
# UTF-16 units in JavaScript, one code point here; both sides hash the same UTF-8 bytes).
WORDS = ["caf" + chr(0xE9), chr(0x2014), "na" + chr(0xEF) + "ve", chr(0x2713), chr(0x2028)]
AWKWARD = " ".join([*WORDS, chr(0x1F680)])


@pytest.mark.parametrize(
    ("secret", "text", "node_signature"),
    [
        (
            "test-signing-secret",
            "Use streamText (Source 1).",
            "v1.1XfOoRXIZhhAmg0tSdHqTepO9h5ouMGLxTk8WXEDDvM",
        ),
        ("test-signing-secret", AWKWARD, "v1.uQUIn8qLEqhG3ATWw91NAHnXiuK1eq3Dvmn9_Dt658o"),
        ("another-secret", "", "v1.MQgR7N_BCUN5baqvWliE7RrjQYMAu3TS_icrfu5OGMc"),
    ],
)
def test_matches_node(secret: str, text: str, node_signature: str) -> None:
    assert sign_assistant_text(text, secret) == node_signature


def test_no_padding_and_url_safe_alphabet() -> None:
    # 32 bytes of HMAC are 43 base64 characters plus one "=", which base64url drops.
    sig = sign_assistant_text("x", "s")
    assert sig.startswith("v1.") and len(sig) == 3 + 43
    assert not set(sig[3:]) & set("+/=")


def test_the_secret_and_the_text_both_matter() -> None:
    base = sign_assistant_text("text", "secret")
    assert sign_assistant_text("text ", "secret") != base
    assert sign_assistant_text("text", "secret2") != base
