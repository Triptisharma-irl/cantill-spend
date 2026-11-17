"""Low-level parsing: amounts, rates, dates. Everything is vectorised over pandas Series of strings."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CURRENCY_JUNK = re.compile(r"[^\d,.\-()+eE]")


def parse_amount(s: pd.Series, decimal: str | None = None) -> pd.Series:
    """Parse messy money strings: '1,234.56', '1.234,56', '(12.50)', '12.50-', 'R 1 234,50', '1e3'.

    decimal=None auto-detects per value: when both ',' and '.' occur the right-most one is the
    decimal separator; a lone ',' followed by exactly 3 digits (and repeated groups) is a
    thousands separator, otherwise it is a decimal comma.
    """
    s = s.fillna("").astype(str).str.strip()
    neg = s.str.match(r"^\(.*\)$") | s.str.endswith("-") | s.str.match(r"^-")
    t = s.str.replace(_CURRENCY_JUNK, "", regex=True).str.strip("()+-")

    def one(x: str) -> float:
        if x == "":
            return np.nan
        if decimal == ".":
            x = x.replace(",", "")
        elif decimal == ",":
            x = x.replace(".", "").replace(",", ".")
        else:
            has_c, has_d = "," in x, "." in x
            if has_c and has_d:
                if x.rfind(",") > x.rfind("."):
                    x = x.replace(".", "").replace(",", ".")
                else:
                    x = x.replace(",", "")
            elif has_c:
                if re.fullmatch(r"\d{1,3}(,\d{3})+", x):
                    x = x.replace(",", "")
                else:
                    x = x.replace(",", ".")
            elif x.count(".") > 1:  # 1.234.567 -> thousands dots
                x = x.replace(".", "")
        try:
            return float(x)
        except ValueError:
            return np.nan

    uniq = pd.unique(t)
    lut = {u: one(u) for u in uniq}
    out = t.map(lut).astype(float)
    return out.where(~neg, -out.abs())


def parse_rate(s: pd.Series, default: float | None = None) -> pd.Series:
    """Tax rates as '15', '15%', '0.15', '' -> fraction. Values > 1 are percentages."""
    raw = s.fillna("").astype(str).str.strip()
    pct = raw.str.endswith("%")
    v = parse_amount(raw.str.rstrip("%"))
    v = v.where(~(pct | (v > 1)), v / 100.0)
    if default is not None:
        v = v.fillna(default)
    return v


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTHS.update({"sept": 9, "ene": 1, "abr": 4, "ago": 8, "dic": 12, "mai": 5, "okt": 10, "dez": 12, "set": 9, "out": 10, "fev": 2, "avr": 4, "juil": 7, "janv": 1, "fevr": 2, "mars": 3, "juin": 6, "aout": 8, "dec.": 12})


def _infer_day_first(vals: pd.Series) -> bool | None:
    """For a column of a/b/yyyy strings decide whether a is the day. None if undecidable."""
    m = vals.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}$").dropna().astype(int)
    if m.empty:
        return None
    a_gt, b_gt = (m[0] > 12).sum(), (m[1] > 12).sum()
    if a_gt and not b_gt:
        return True
    if b_gt and not a_gt:
        return False
    if a_gt or b_gt:
        return a_gt >= b_gt
    return None


def parse_dates(s: pd.Series, formats: list[str] | None = None, day_first: bool | None = None) -> pd.Series:
    """Parse a column that mixes date formats. Returns datetime64 (NaT where unparseable).

    Tries, in order: explicit formats from config, ISO yyyy-mm-dd[ time], dd-Mon-yy(yy),
    a/b/yyyy with day/month order inferred from the column, yyyymmdd, Excel serials."""
    raw = s.fillna("").astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    todo = raw != ""

    def fill(mask, parsed):
        nonlocal out, todo
        ok = mask & parsed.notna() & out.isna()
        out = out.where(~ok, parsed)
        todo = todo & out.isna()

    for fmt in formats or []:
        fill(todo, pd.to_datetime(raw.where(todo), format=fmt, errors="coerce"))

    iso = raw.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    fill(todo & iso, pd.to_datetime(raw.where(todo & iso).str[:10], format="%Y-%m-%d", errors="coerce"))

    # 05-Mar-24 / 5 Mar 2024 / 05.Mar.2024 (month names, several languages)
    mon = raw.str.extract(r"^(\d{1,2})[\s\-/.]+([A-Za-zéû]{3,5})\.?[\s\-/.,]+(\d{2,4})$")
    has_mon = mon[0].notna() & todo
    if has_mon.any():
        mm = mon[1].str.lower().map(_MONTHS)
        yy = pd.to_numeric(mon[2], errors="coerce")
        yy = yy.where(yy >= 100, 2000 + yy)
        parsed = pd.to_datetime(dict(year=yy, month=mm, day=pd.to_numeric(mon[0], errors="coerce")), errors="coerce")
        fill(has_mon, parsed)

    # Mon 5, 2024
    mon2 = raw.str.extract(r"^([A-Za-z]{3,5})\.?\s+(\d{1,2}),?\s+(\d{4})$")
    has_mon2 = mon2[0].notna() & todo
    if has_mon2.any():
        parsed = pd.to_datetime(dict(year=pd.to_numeric(mon2[2], errors="coerce"), month=mon2[0].str.lower().map(_MONTHS),
                                     day=pd.to_numeric(mon2[1], errors="coerce")), errors="coerce")
        fill(has_mon2, parsed)

    slash = raw.str.match(r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}$") & todo
    if slash.any():
        df = day_first if day_first is not None else _infer_day_first(raw[slash])
        df = True if df is None else df  # most non-US ERPs are day-first; recorded in DECISIONS.md
        parts = raw[slash].str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$").astype(int)
        yy = parts[2].where(parts[2] >= 100, 2000 + parts[2])
        d, m = (parts[0], parts[1]) if df else (parts[1], parts[0])
        fill(slash, pd.to_datetime(dict(year=yy, month=m, day=d), errors="coerce").reindex(raw.index))

    compact = raw.str.match(r"^(19|20)\d{6}$") & todo
    fill(compact, pd.to_datetime(raw.where(compact), format="%Y%m%d", errors="coerce"))

    serial = raw.str.match(r"^\d{5}(\.\d+)?$") & todo
    if serial.any():
        n = pd.to_numeric(raw.where(serial), errors="coerce")
        n = n.where((n > 20000) & (n < 60000))
        fill(serial, pd.to_datetime("1899-12-30") + pd.to_timedelta(n, unit="D"))

    # last resort: let pandas guess element-wise
    if todo.any():
        guess = pd.to_datetime(raw.where(todo), errors="coerce", format="mixed", dayfirst=True)
        fill(todo, guess)
    return out


def date_flags(raw: pd.Series, parsed: pd.Series, valid_from: str | None, valid_to: str | None) -> pd.Series:
    """'' ok, 'MISSING', 'UNPARSEABLE', 'OUT_OF_RANGE'."""
    raw = raw.fillna("").astype(str).str.strip()
    flag = pd.Series("", index=raw.index, dtype=object)
    flag[raw == ""] = "MISSING"
    flag[(raw != "") & parsed.isna()] = "UNPARSEABLE"
    if valid_from or valid_to:
        lo = pd.Timestamp(valid_from) if valid_from else pd.Timestamp.min
        hi = pd.Timestamp(valid_to) if valid_to else pd.Timestamp.max
        flag[parsed.notna() & ((parsed < lo) | (parsed > hi))] = "OUT_OF_RANGE"
    return flag
