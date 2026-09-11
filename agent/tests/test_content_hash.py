"""hash_chunk must produce the same bytes as lib/content-hash.ts and db/001_content_hash.sql."""

import hashlib

from copilot_agent.content_hash import hash_chunk


def test_pins_the_value_the_typescript_test_pins() -> None:
    # The same literal as tests/content-hash.test.ts. Both sides pinned to one value is what
    # makes them provably equal, without either test running the other language.
    assert (
        hash_chunk("u", "c") == "f83e30bab19727a0baa8f3310891dcd0f7fdb9cbea7154c6905cc4aac96e0c9d"
    )


def test_hashes_url_lf_content_as_utf8() -> None:
    url, content = "https://ai-sdk.dev/docs/x.md", "streamText → café"
    expected = hashlib.sha256(f"{url}\n{content}".encode()).hexdigest()
    assert hash_chunk(url, content) == expected


def test_same_text_on_two_pages_gets_two_keys() -> None:
    assert hash_chunk("https://a", "same") != hash_chunk("https://b", "same")


def test_separator_matters() -> None:
    assert hash_chunk("u", "c") != hashlib.sha256(b"uc").hexdigest()
