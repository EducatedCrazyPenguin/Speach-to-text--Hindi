from __future__ import annotations

import ctypes
from ctypes import wintypes
import os


DEFAULT_TARGET = "PrivateConversationTranscriber/HuggingFace"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168


class _CredentialW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(wintypes.BYTE)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


_CredentialPointer = ctypes.POINTER(_CredentialW)


def _credential_api():
    if os.name != "nt":
        raise RuntimeError("Secure token storage is available only on Windows.")
    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    advapi32.CredWriteW.argtypes = [ctypes.POINTER(_CredentialW), wintypes.DWORD]
    advapi32.CredWriteW.restype = wintypes.BOOL
    advapi32.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_CredentialPointer),
    ]
    advapi32.CredReadW.restype = wintypes.BOOL
    advapi32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    advapi32.CredDeleteW.restype = wintypes.BOOL
    advapi32.CredFree.argtypes = [ctypes.c_void_p]
    advapi32.CredFree.restype = None
    return advapi32


def save_token(token: str, target: str = DEFAULT_TARGET) -> None:
    token_bytes = token.strip().encode("utf-16-le")
    if not token_bytes:
        raise ValueError("Cannot save an empty token.")
    blob = (wintypes.BYTE * len(token_bytes)).from_buffer_copy(token_bytes)
    credential = _CredentialW(
        Flags=0,
        Type=_CRED_TYPE_GENERIC,
        TargetName=target,
        Comment="Private Conversation Transcriber Hugging Face read token",
        CredentialBlobSize=len(token_bytes),
        CredentialBlob=ctypes.cast(blob, ctypes.POINTER(wintypes.BYTE)),
        Persist=_CRED_PERSIST_LOCAL_MACHINE,
        AttributeCount=0,
        Attributes=None,
        TargetAlias=None,
        UserName="HuggingFace",
    )
    api = _credential_api()
    if not api.CredWriteW(ctypes.byref(credential), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def load_token(target: str = DEFAULT_TARGET) -> str:
    api = _credential_api()
    credential = _CredentialPointer()
    if not api.CredReadW(target, _CRED_TYPE_GENERIC, 0, ctypes.byref(credential)):
        error = ctypes.get_last_error()
        if error == _ERROR_NOT_FOUND:
            return ""
        raise ctypes.WinError(error)
    try:
        value = ctypes.string_at(
            credential.contents.CredentialBlob,
            credential.contents.CredentialBlobSize,
        )
        return value.decode("utf-16-le")
    finally:
        api.CredFree(credential)


def has_token(target: str = DEFAULT_TARGET) -> bool:
    try:
        return bool(load_token(target))
    except OSError:
        return False


def forget_token(target: str = DEFAULT_TARGET) -> None:
    api = _credential_api()
    if not api.CredDeleteW(target, _CRED_TYPE_GENERIC, 0):
        error = ctypes.get_last_error()
        if error != _ERROR_NOT_FOUND:
            raise ctypes.WinError(error)
