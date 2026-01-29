# dambae_gate.py
import re

def _normalize(text: str) -> str:
    """
    - 공백 제거
    - 한글만 남김(구두점/기호/영문/숫자 제거)
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^가-힣]", "", text)
    return text

def is_only_dambae(stt_text: str) -> bool:
    """
    오직 '담배'만 통과.
    예)
      '담배'       -> True
      '담배.'      -> True (구두점 제거)
      ' 담 배 '    -> True (공백 제거)
      '담비'       -> False
      '담배요'     -> False
      '담배 가져와' -> False (한글만 남기면 '담배가져와'라서 False)
    """
    return _normalize(stt_text) == "담배"

