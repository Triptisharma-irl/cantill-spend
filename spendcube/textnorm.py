"""Text normalisation shared by matching, de-duplication and classification."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# OCR confusions: digits that sit inside alphabetic words are almost always misread letters.
_OCR_IN_WORD = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "4": "A", "3": "E", "6": "G", "7": "T", "2": "Z"})


def _ocr_fix_token(tok: str) -> str:
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    if letters < 3 or digits == 0 or digits > max(1, letters // 3) or re.search(r"\d{2,}", tok):
        return tok
    chars = list(tok)
    for i, c in enumerate(chars):
        if not c.isdigit():
            continue
        left = tok[:i]
        right = tok[i + 1:]
        interior = bool(left) and bool(right) and left[-1].isalpha() and right[0].isalpha()
        lead = not left and re.match(r"[A-Za-z]{4,}$", right) is not None
        trail = not right and re.search(r"[A-Za-z]{4,}$", left) is not None and len(left) >= 4
        if interior or lead or trail:
            chars[i] = c.translate(_OCR_IN_WORD)
    return "".join(chars)


def ocr_fix(text: str) -> str:
    return re.sub(r"[A-Za-z0-9]+", lambda m: _ocr_fix_token(m.group(0)), text)


def clean_text(text: str) -> str:
    """Upper-case, accent-free, OCR-repaired, single-spaced."""
    if not isinstance(text, str):
        return ""
    t = strip_accents(text).upper()
    t = ocr_fix(t)
    t = re.sub(r"\s+", " ", t).strip(" -;,.")
    return t


def compact(text: str) -> str:
    """Key that ignores spacing/punctuation: 'ROD ANCHR;DIA:20MM' -> 'RODANCHRDIA20MM'."""
    return re.sub(r"[^A-Z0-9]", "", clean_text(text))


class SupplierNormaliser:
    def __init__(self, legal_suffixes: list[str]):
        self.suffixes = {s.upper().replace(".", "") for s in legal_suffixes}

    def tokens(self, name: str) -> list[str]:
        t = clean_text(name).replace("&", " AND ")
        t = re.sub(r"\(.*?\)", lambda m: " " + m.group(0)[1:-1] + " ", t)
        toks = re.findall(r"[A-Z0-9]+", t.replace(".", ""))
        core = [x for x in toks if x not in self.suffixes]
        return core or toks

    def key(self, name: str) -> str:
        return "".join(self.tokens(name))


def tokenize(text: str, abbreviations: dict[str, str] | None = None) -> list[str]:
    """Lower-case word tokens with abbreviation expansion, for classification features."""
    if not isinstance(text, str) or not text:
        return []
    t = strip_accents(text).lower()
    t = re.sub(r"[_/;:,()\[\]{}|+*#=\"'<>!?\\]", " ", t)
    toks = re.findall(r"[a-zЀ-ӿͰ-Ͽ一-鿿]+|\d+(?:[.,]\d+)?", t)
    if abbreviations:
        out = []
        for x in toks:
            e = abbreviations.get(x)
            if e is None:
                out.append(x)
            elif e:
                out.extend(e.split())
        toks = out
    return toks


def is_generic(desc: str, generic: set[str]) -> bool:
    t = re.sub(r"[^a-z0-9 ]", " ", strip_accents(desc or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t in generic or len(re.sub(r"[^a-z]", "", t)) < 2


def series_map(s: pd.Series, fn) -> pd.Series:
    """Apply fn once per unique value (data has heavy repetition)."""
    u = pd.unique(s.fillna(""))
    lut = {v: fn(v) for v in u}
    return s.fillna("").map(lut)
