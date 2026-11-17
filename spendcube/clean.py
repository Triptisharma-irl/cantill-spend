"""Amounts to net reporting currency; dates parsed and flagged."""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .ingest import Config, read_csv
from .parse import date_flags, parse_amount, parse_dates

log = logging.getLogger("spendcube")


class DataError(ValueError):
    """The data cannot be processed as configured (the message says what to change)."""


def load_fx_table(cfg: Config, data_dir: str, train_dir: str | None) -> pd.DataFrame | None:
    spec = cfg.get("fx_table")
    if not spec:
        return None
    for d in [data_dir, train_dir]:
        if d and os.path.exists(os.path.join(d, spec["file"])):
            t = read_csv(os.path.join(d, spec["file"]))
            out = pd.DataFrame({
                "currency": t[spec.get("currency", "currency")].str.upper().str.strip(),
                "month": pd.to_datetime(t[spec.get("month", "month")].str.strip().str[:7], format="%Y-%m", errors="coerce"),
                "rate": parse_amount(t[spec.get("rate", "rate")]),
            }).dropna()
            if spec.get("semantics", "units_per_reporting") == "reporting_per_unit":
                out["rate"] = 1.0 / out["rate"]
            return out.sort_values(["currency", "month"])
    return None


def _table_rate(fx: pd.DataFrame, cur: pd.Series, when: pd.Series) -> pd.Series:
    """Rate for the invoice month; nearest available month of that currency if the month is missing."""
    res = pd.Series(np.nan, index=cur.index)
    month = when.dt.to_period("M").dt.to_timestamp()
    for c, g in fx.groupby("currency"):
        m = (cur == c)
        if not m.any():
            continue
        lut = dict(zip(g["month"], g["rate"]))
        exact = month[m].map(lut)
        missing = exact.isna()
        if missing.any():
            months = g["month"].values.astype("datetime64[ns]")
            rates = g["rate"].values
            tgt = month[m][missing]
            # nearest month (undated lines get the median rate of that currency)
            idx = np.full(len(tgt), -1)
            valid = tgt.notna().values
            if valid.any():
                tv = tgt[valid].values.astype("datetime64[ns]").astype("int64")
                mv = months.astype("int64")
                idx[valid] = np.abs(tv[:, None] - mv[None, :]).argmin(axis=1)
            fill = np.where(idx >= 0, rates[np.clip(idx, 0, None)], np.median(rates))
            exact[missing] = fill
        res[m] = exact.values
    return res


def clean(df: pd.DataFrame, cfg: Config, fx: pd.DataFrame | None) -> pd.DataFrame:
    rep = cfg["reporting_currency"].upper()
    dates_cfg = cfg.get("dates", {})
    df = df.copy()
    # ---- dates -------------------------------------------------------------
    df["date"] = parse_dates(df["date_raw"], dates_cfg.get("formats"), dates_cfg.get("day_first"))
    df["date_flag"] = date_flags(df["date_raw"], df["date"], dates_cfg.get("valid_from"), dates_cfg.get("valid_to"))
    fxd = parse_dates(df["fx_date_raw"], dates_cfg.get("formats"), dates_cfg.get("day_first"))
    fxd_flag = date_flags(df["fx_date_raw"], fxd, dates_cfg.get("valid_from"), dates_cfg.get("valid_to"))
    fb = parse_dates(df["fx_date_fallback_raw"], dates_cfg.get("formats"), dates_cfg.get("day_first"))
    use_fb = (fxd_flag != "")
    df["fx_date"] = fxd.where(~use_fb, fb)
    df["fx_date_flag"] = np.where(use_fb, "FALLBACK:" + fxd_flag.astype(str), "")
    # ---- net of tax --------------------------------------------------------
    gross = df["basis"].eq("gross")
    df["amount_net_doc"] = np.where(gross, df["amount_raw"] / (1.0 + df["tax_rate"].fillna(0.0)), df["amount_raw"])
    # ---- currency ----------------------------------------------------------
    cur = df["currency"].replace("", rep)
    df["currency"] = cur
    is_rep = cur.eq(rep)
    line_rate = df["line_fx_rate"].astype(float)
    line_rate = line_rate.where(line_rate > 0)
    rate = pd.Series(np.where(is_rep, 1.0, np.nan), index=df.index)
    src_note = pd.Series(np.where(is_rep, "REPORTING", ""), index=df.index, dtype=object)
    tbl = _table_rate(fx, cur, df["fx_date"]) if fx is not None else pd.Series(np.nan, index=df.index)
    max_dev = cfg.get("fx_line_rate_max_deviation", 3.0)
    ok_line = (~is_rep) & line_rate.notna()
    sane = ok_line & (tbl.isna() | ((line_rate / tbl).between(1.0 / max_dev, max_dev)))
    rate = rate.where(~sane, line_rate)
    src_note[sane] = "LINE_RATE"
    use_tbl = (~is_rep) & rate.isna() & tbl.notna()
    rate = rate.where(~use_tbl, tbl)
    src_note[use_tbl & ok_line] = "TABLE_RATE(line rate implausible)"
    src_note[use_tbl & ~ok_line] = "TABLE_RATE"
    src_note[(~is_rep) & rate.isna()] = "NO_RATE"
    df["fx_rate"] = rate
    df["fx_source"] = src_note
    df["spend_all"] = (df["amount_net_doc"] / df["fx_rate"]).round(2)
    df["amount_flag"] = np.where(df["amount_raw"].isna(), "UNPARSEABLE_AMOUNT", np.where(df["fx_rate"].isna(), "NO_FX_RATE", ""))
    df["spend_all"] = df["spend_all"].fillna(0.0)
    _check_fx_coverage(df, rep, cfg.get("max_unconverted_share", 0.01))
    return df


def _check_fx_coverage(df: pd.DataFrame, rep: str, max_share: float) -> None:
    """Never write silent zeros: stop when a material share of spend could not be converted."""
    miss = df["amount_flag"].eq("NO_FX_RATE")
    if not miss.any():
        return
    amt = df["amount_net_doc"].abs().fillna(0.0)
    share = amt[miss].sum() / max(amt.sum(), 1e-9)
    by = df.loc[miss, "currency"].value_counts().head(8).to_dict()
    msg = (f"{miss.sum()} lines ({share:.1%} of document-currency amount) have no FX rate to {rep}; "
           f"currencies: {by}. Unconverted lines get spend 0.")
    if share > max_share:
        raise DataError(msg + f" Above max_unconverted_share={max_share}: check reporting_currency, add an "
                         "fx_table to the config, or set max_unconverted_share to accept it.")
    log.warning(msg)
