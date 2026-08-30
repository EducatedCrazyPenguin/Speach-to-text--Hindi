from __future__ import annotations

import os
import uuid

import pytest

from voice_to_text.secrets import forget_token, load_token, save_token


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI test")
def test_token_is_stored_in_windows_credential_manager() -> None:
    target = f"PrivateConversationTranscriber/Test/{uuid.uuid4().hex}"
    token = "hf_test_token_that_must_not_appear_in_plaintext"
    try:
        try:
            save_token(token, target)
        except OSError as exc:
            if exc.winerror == 1312:
                pytest.skip("Windows Credential Manager is unavailable in this logon session")
            raise
        assert load_token(target) == token
    finally:
        forget_token(target)
    assert load_token(target) == ""
