"""Classify spend lines into a category tree read at run time.

Signals (combined in probability space over the tree's leaves):
  1. zero-shot: similarity of each line to each leaf path (multilingual embeddings + lexical TF-IDF
     over abbreviation-expanded tokens). Needs only the tree.
  2. transductive smoothing: lines of the same supplier / item / GL account tend to share a category,
     so each line borrows evidence from its group.
  3. self-training: a linear model trained on the most confident pseudo-labels generalises from
     descriptions it can read to ones it cannot (abbreviations, other languages, codes).
  4. supervised (only when the company provides labels): linear model + exact-match memory.
     Labels are optional; with none the tool runs on 1-3.
The final category is chosen top-down (L1 by summed probability, then L2 inside it, ...), which
maximises per-level accuracy, and the confidence is the probability mass of the chosen leaf.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from ..taxonomy import Taxonomy
from ..textnorm import clean_text, compact, is_generic, tokenize
from .embed import Embedder

log = logging.getLogger("spendcube")

_STOP = set("""a an and or of for the to in on at by with from per as is are be into via de del la le les el los las
da do dos das du des et y e en und der die das von zu mit fur pour par con sin para com sem""".split())


def _stem(w: str) -> str:
    if len(w) <= 3 or not w.isalpha():
        return w
    for suf in ("ies", "ing", "ers", "es", "ed", "er", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    return w


def _analyzer(abbrev):
    def an(text: str):
        toks = [t for t in tokenize(text, abbrev) if t not in _STOP]
        toks = [_stem(t) for t in toks if not t.isdigit() and len(t) > 1]
        return toks
    return an


def _softmax(S: np.ndarray, T: float) -> np.ndarray:
    Z = (S - S.max(axis=1, keepdims=True)) / T
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _renorm(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)


@dataclass
class ClassifierConfig:
    use_embeddings: bool = True
    embed_model: str | None = None
    temperature: float = 0.04
    w_embed: float = 0.6
    w_lex_word: float = 0.25
    w_lex_char: float = 0.15
    w_supplier_name: float = 0.15
    group_weights: dict = field(default_factory=lambda: {"supplier_key": 0.6, "item_key": 0.8, "gl_desc": 0.3})
    self_training_rounds: int = 1
    self_training_frac: float = 0.5
    w_self_train: float = 0.4
    supervised_C: float = 4.0
    min_memory_count: int = 1
    llm: str = "auto"            # auto: use when no labels and credentials exist; off: never
    llm_model: str | None = None
    llm_weight: float = 0.75
    llm_max_units: int = 8000
    # an item key whose described lines spread over many leaves is a catch-all code, not an item
    catchall_min_texts: int = 8
    catchall_max_purity: float = 0.5
    # lines with no readable description borrow from described lines of the same supplier/GL/context
    backfill_weights: dict = field(default_factory=lambda: {"supplier_key": 1.0, "item_key": 1.0, "gl_desc": 0.5,
                                                             "context": 0.5})
    backfill_prior: float = 0.2


class SpendClassifier:
    def __init__(self, tax: Taxonomy, abbreviations: dict, ccfg: ClassifierConfig | None = None,
                 generic_descriptions=()):
        self.tax = tax
        self.abbrev = abbreviations
        self.generic = set(generic_descriptions)
        self.c = ccfg or ClassifierConfig()
        self.embedder = Embedder(self.c.embed_model) if (self.c.use_embeddings and self.c.embed_model) else (
            Embedder() if self.c.use_embeddings else None)
        self.leaf_texts = [self._leaf_text(p) for p in tax.leaves]

    # ------------------------------------------------------------------ text
    def _leaf_text(self, path: tuple) -> str:
        # leaf name carries most weight; ancestors give context
        names = list(path)
        txt = " ; ".join(names[::-1])
        d = self.tax.descriptions.get(path)
        return f"{txt} ; {d}" if d else txt

    @staticmethod
    def unit_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Collapse lines to unique classification units."""
        keycols = ["description", "supplier_key", "supplier_name", "gl_desc", "context", "item_key"]
        k = df[keycols].fillna("").astype(str).agg("\x1f".join, axis=1)
        codes, uniq = pd.factorize(k)
        U = df.groupby(codes).agg(**{c: (c, "first") for c in keycols},
                                  n=("description", "size"), spend=("spend_abs", "sum")).reset_index(drop=True)
        return U, codes

    def _texts(self, U: pd.DataFrame):
        desc = U["description"].fillna("").map(clean_text)
        exp = desc.map(lambda t: " ".join(tokenize(t, self.abbrev)))
        sup = U["supplier_name"].fillna("").map(clean_text)
        ctx = U["context"].fillna("").str.replace(r"\b\w+=", " ", regex=True)
        full = exp + " | " + sup.str.lower() + " | " + U["gl_desc"].fillna("").str.lower() + " | " + ctx.str.lower()
        return desc, exp, sup, full

    # ------------------------------------------------------------------ zero-shot
    def zero_shot(self, U: pd.DataFrame) -> np.ndarray:
        desc, exp, sup, full = self._texts(U)
        L = self.tax.n
        S = np.zeros((len(U), L))
        # lexical, word level: leaf path vs expanded description (+ gl/context, weaker)
        an = _analyzer(self.abbrev)
        leaf_docs = [" ".join(p) + " " + p[-1] for p in self.tax.leaves]  # leaf name counted twice
        vw = TfidfVectorizer(analyzer=an, sublinear_tf=True, min_df=1)
        vw.fit(list(exp) + leaf_docs)
        Xw = normalize(vw.transform(full))
        Lw = normalize(vw.transform(leaf_docs))
        Sw = (Xw @ Lw.T).toarray()
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        vc.fit(list(exp) + leaf_docs)
        Sc = (normalize(vc.transform(exp)) @ normalize(vc.transform(leaf_docs)).T).toarray()
        S += self.c.w_lex_word * Sw + self.c.w_lex_char * Sc
        wsum = self.c.w_lex_word + self.c.w_lex_char
        if self.embedder is not None:
            leaf_variants = [[p[-1], " ".join(p[-2:]), " > ".join(p)] for p in self.tax.leaves]
            flat = [v for vs in leaf_variants for v in vs]
            LE = self.embedder.encode(flat)
            if LE is not None:
                LE = LE.reshape(L, 3, -1).mean(axis=1)
                LE /= np.linalg.norm(LE, axis=1, keepdims=True)
                # embed the human-readable (expanded) description; blank -> supplier name
                text_for_emb = [e if len(e) > 2 else s.lower() for e, s in zip(exp, sup)]
                DE = self.embedder.encode(text_for_emb)
                Se = DE @ LE.T
                Se[np.array([len(t.strip()) <= 2 for t in text_for_emb])] = 0.0  # nothing to read
                S += self.c.w_embed * Se
                wsum += self.c.w_embed
                if self.c.w_supplier_name > 0:
                    SE = self.embedder.encode([s.lower() for s in sup])
                    Ss = SE @ LE.T
                    Ss[(sup.str.len() <= 2).values] = 0.0
                    S += self.c.w_supplier_name * Ss
                    wsum += self.c.w_supplier_name
        S /= wsum
        S += self._acronym_scores(desc)
        # centre per leaf: some leaf names are similar to everything ("services", "other")
        S = S - 0.5 * S.mean(axis=0, keepdims=True)
        return _softmax(S, self.c.temperature)

    def _acronym_scores(self, desc: pd.Series, bonus: float = 0.1) -> np.ndarray:
        """'CBS 1783.9 X 0.245' -> leaf 'Can Body Stock (coil)'. Acronyms = initials of a node name (>=2 words)."""
        acr = {}
        for i, p in enumerate(self.tax.leaves):
            for lvl, name in enumerate(p):
                words = [w for w in re.findall(r"[A-Za-z]+", re.sub(r"\(.*?\)", " ", name)) if w.lower() not in _STOP]
                if len(words) >= 2:
                    a = "".join(w[0] for w in words).upper()
                    if len(a) >= 2:
                        acr.setdefault(a, set()).add((i, 1.0 if lvl == len(p) - 1 else 0.5))
        S = np.zeros((len(desc), self.tax.n))
        if not acr:
            return S
        for r, t in enumerate(desc.values):
            for tok in set(re.findall(r"\b[A-Z]{2,5}\b", t)):
                for i, w in acr.get(tok, ()):
                    S[r, i] = max(S[r, i], bonus * w)
        return S

    # ------------------------------------------------------------------ smoothing
    def smooth(self, U: pd.DataFrame, P: np.ndarray, weights: dict | None = None) -> np.ndarray:
        weights = self.c.group_weights if weights is None else weights
        out = P.copy()
        w = U["n"].values.astype(float)[:, None]
        for col, gw in weights.items():
            if gw <= 0 or col not in U:
                continue
            g = U[col].fillna("").astype(str).values
            codes, uniq = pd.factorize(g)
            sums = np.zeros((len(uniq), P.shape[1]))
            np.add.at(sums, codes, P * w)
            cnt = np.zeros(len(uniq))
            np.add.at(cnt, codes, w[:, 0])
            G = sums / cnt[:, None]
            blank = np.array([u == "" for u in uniq])
            # group evidence excluding the unit itself (leave-one-out) so singletons add nothing
            own = P * w
            Gl = (sums[codes] - own) / np.maximum(cnt[codes] - w[:, 0], 1e-9)[:, None]
            has_others = (cnt[codes] - w[:, 0]) > 0
            Gl[~has_others] = P[~has_others]
            Gl[blank[codes]] = P[blank[codes]]
            out = out + gw * Gl
        return _renorm(out)

    def informative(self, U: pd.DataFrame) -> np.ndarray:
        """Units whose description says something (not blank, not 'product', not a bare code)."""
        d = U["description"].fillna("")
        return ~d.map(lambda t: is_generic(t, self.generic)).values

    def mask_catchall_items(self, U: pd.DataFrame, P: np.ndarray, informative: np.ndarray) -> pd.DataFrame:
        """Blank out item keys that are catch-all codes: their described lines are many different
        texts that point at different leaves, so the key carries no category information."""
        item = U["item_key"].fillna("").astype(str)
        d = pd.DataFrame({"item": item, "leaf": P.argmax(axis=1), "w": U["n"].values,
                          "txt": U["description"].fillna("").map(compact)})[informative & (item != "").values]
        if d.empty:
            return U
        ntxt = d.groupby("item")["txt"].nunique()
        lw = d.groupby(["item", "leaf"])["w"].sum()
        purity = lw.groupby(level=0).max() / lw.groupby(level=0).sum()
        bad = set(purity[(purity < self.c.catchall_max_purity) & (ntxt.reindex(purity.index) >= self.c.catchall_min_texts)].index)
        if bad:
            n = item.isin(bad)
            log.info("catch-all item keys ignored for classification: %d keys covering %d units", len(bad), int(n.sum()))
            U = U.copy()
            U.loc[n.values, "item_key"] = ""
        return U

    def backfill(self, U: pd.DataFrame, P: np.ndarray, informative: np.ndarray) -> np.ndarray:
        """Units with nothing readable take their evidence from described units of the same supplier,
        item (catch-all codes already removed), GL account and context, then from the overall
        spend-weighted category mix."""
        empty = ~informative
        if not empty.any() or not informative.any():
            return P
        acc = np.zeros_like(P)
        donors = informative.astype(float)
        w = U["n"].values.astype(float)[:, None] * donors[:, None]
        for col, gw in self.c.backfill_weights.items():
            if gw <= 0 or col not in U:
                continue
            g = U[col].fillna("").astype(str).values
            codes, uniq = pd.factorize(g)
            sums = np.zeros((len(uniq), P.shape[1]))
            np.add.at(sums, codes, P * w)
            cnt = sums.sum(axis=1)
            ok = (cnt[codes] > 0) & (np.array([u != "" for u in uniq])[codes])
            G = np.zeros_like(P)
            G[ok] = sums[codes[ok]] / cnt[codes[ok], None]
            acc += gw * G
        sw = (U["spend"].abs().values + 1e-9)[:, None] * informative[:, None]
        prior = (P * sw).sum(axis=0) / sw.sum()
        acc += self.c.backfill_prior * prior[None, :]
        out = P.copy()
        out[empty] = _renorm(0.2 * P[empty] + _renorm(acc[empty]))
        return out

    # ------------------------------------------------------------------ features for linear models
    def _feature_space(self, U: pd.DataFrame):
        desc, exp, sup, full = self._texts(U)
        an = _analyzer(self.abbrev)
        self.vw = TfidfVectorizer(analyzer=an, ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=200000)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=300000)
        meta = ("sup_" + U["supplier_key"].fillna("").astype(str).str.replace(r"\W", "_", regex=True) + " gl_" +
                U["gl_desc"].fillna("").astype(str).str.replace(r"\W", "_", regex=True) + " item_" +
                U["item_key"].fillna("").astype(str).str.replace(r"\W", "_", regex=True) + " " +
                U["context"].fillna("").astype(str).str.replace(r"[^\w=|]", "_", regex=True).str.replace("|", " ", regex=False))
        self.vm = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=True, min_df=1)
        Xw = self.vw.fit_transform(exp)
        Xc = self.vc.fit_transform(desc)
        Xm = self.vm.fit_transform(meta)
        return sp.hstack([Xw, Xc * 0.8, Xm * 0.7]).tocsr()

    def _fit_predict(self, X, y, w, Xall, n_classes, C):
        classes = np.unique(y)
        if len(classes) < 2:
            P = np.zeros((Xall.shape[0], n_classes))
            P[:, classes[0]] = 1.0
            return P
        clf = LogisticRegression(C=C, max_iter=300, solver="saga", tol=1e-3, random_state=0)  # reproducible runs
        clf.fit(X, y, sample_weight=w)
        Pc = clf.predict_proba(Xall)
        P = np.full((Xall.shape[0], n_classes), 1e-6)
        P[:, clf.classes_] = Pc
        return _renorm(P)

    def self_train(self, U: pd.DataFrame, P: np.ndarray, X) -> np.ndarray:
        cur = P
        for r in range(self.c.self_training_rounds):
            conf = cur.max(axis=1)
            lab = cur.argmax(axis=1)
            # take the most confident fraction per predicted class (avoids big classes swamping)
            keep = np.zeros(len(U), bool)
            for k in np.unique(lab):
                idx = np.where(lab == k)[0]
                thr = np.quantile(conf[idx], 1 - self.c.self_training_frac)
                keep[idx[conf[idx] >= thr]] = True
            w = conf[keep] * np.log1p(U["n"].values[keep])
            Pst = self._fit_predict(X[keep], lab[keep], w, X, self.tax.n, C=2.0)
            cur = _renorm((1 - self.c.w_self_train) * P + self.c.w_self_train * Pst)
            log.info("self-training round %d: %d pseudo-labelled units", r + 1, keep.sum())
        return cur

    # ------------------------------------------------------------------ main entry
    def classify(self, lines: pd.DataFrame, train: pd.DataFrame | None = None) -> pd.DataFrame:
        """lines: description, supplier_key, supplier_name, gl_desc, context, item_key, spend_abs.
        train (optional): same columns + 'leaf' (index into taxonomy leaves)."""
        allx = lines if train is None else pd.concat([lines, train], ignore_index=True)
        U, codes = self.unit_frame(allx)
        n_lines = len(lines)
        log.info("classifying %d lines as %d unique units into %d leaves", n_lines, len(U), self.tax.n)
        P0 = self.zero_shot(U)
        inf = self.informative(U)
        seen = np.zeros(len(U), bool)  # units with a known label are never back-filled
        if train is not None and len(train):
            tcodes = codes[n_lines:]
            y = train["leaf"].values
            Pl = P0.copy()
            Pl[tcodes] = np.eye(self.tax.n)[y]  # labels decide what an item key holds, where known
            U = self.mask_catchall_items(U, Pl, inf)
        else:
            U = self.mask_catchall_items(U, P0, inf)
        P1 = self.backfill(U, self.smooth(U, P0), inf)
        X = self._feature_space(U)
        if train is not None and len(train):
            # memory: label distribution per unit
            M = np.zeros((len(U), self.tax.n))
            np.add.at(M, tcodes, np.eye(self.tax.n)[y])
            seen = M.sum(axis=1) > 0
            lab_units = np.where(seen)[0]
            ylab = M[lab_units].argmax(axis=1)
            wlab = M[lab_units].sum(axis=1)
            Psup = self._fit_predict(X[lab_units], ylab, np.log1p(wlab) + 1, X, self.tax.n, C=self.c.supervised_C)
            Psup = self.smooth(U, Psup, {"item_key": 0.3, "supplier_key": 0.1})
            # trust supervised more where the line looks like training data
            sim = np.asarray((X @ X[lab_units].T).max(axis=1).todense()).ravel() if len(lab_units) < 60000 else np.ones(len(U))
            coverage = min(1.0, len(train) / max(n_lines, 1))
            alpha = np.clip(0.55 + 0.45 * sim, 0, 0.97)[:, None]
            alpha = alpha * (0.6 + 0.4 * coverage)
            P = _renorm(alpha * Psup + (1 - alpha) * P1)
            Mem = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
            P[seen] = _renorm(0.85 * Mem[seen] + 0.15 * P[seen])
        else:
            P = self.self_train(U, P1, X) if self.c.self_training_rounds else P1
            P = self._with_llm(U, P)
        P = self.backfill(U, P, inf | seen)
        return self._decide(P[codes[:n_lines]])

    def _with_llm(self, U: pd.DataFrame, P: np.ndarray) -> np.ndarray:
        from . import llm
        if self.c.llm == "off" or not llm.available():
            if self.c.llm != "off":
                log.info("LLM not configured (no Anthropic credentials) -> local classifier only")
            return P
        try:
            lc = llm.LLMClassifier(self.tax.leaves, model=self.c.llm_model, max_units=self.c.llm_max_units)
            ctx = U["context"].fillna("").str.replace(r"\b\w+=", "", regex=True)
            texts = ("desc: " + U["description"].fillna("").str.strip() + " | supplier: " + U["supplier_name"].fillna("") +
                     " | gl: " + U["gl_desc"].fillna("") + " | " + ctx).tolist()
            PL = lc.classify(texts, U["spend"].values)
        except Exception as e:  # never let the optional backend break a run
            log.warning("LLM classifier unavailable (%s) -> local only", e)
            return P
        has = PL.sum(axis=1) > 0
        log.info("LLM labels available for %d of %d units", has.sum(), len(U))
        out = P.copy()
        out[has] = _renorm((1 - self.c.llm_weight) * P[has] + self.c.llm_weight * PL[has])
        # spread LLM knowledge to units it did not see (same supplier / item)
        return self.smooth(U, out, {"supplier_key": 0.3, "item_key": 0.6})

    def _decide(self, P: np.ndarray) -> pd.DataFrame:
        leaves = self.tax.leaves
        depth = self.tax.depth
        n = P.shape[0]
        chosen = np.arange(self.tax.n)[None, :].repeat(n, axis=0)
        mask = np.ones_like(P, dtype=bool)
        for lvl in range(1, depth + 1):
            keys = [p[:lvl] if len(p) >= lvl else p for p in leaves]
            codes, uniq = pd.factorize(pd.Series(["\x1f".join(k) for k in keys]))
            M = np.zeros((self.tax.n, len(uniq)))
            M[np.arange(self.tax.n), codes] = 1
            mass = (P * mask) @ M
            best = mass.argmax(axis=1)
            mask &= (codes[None, :] == best[:, None])
        leaf = (P * mask).argmax(axis=1)
        conf = P[np.arange(n), leaf]
        # confidence also reflects how sure the top level is
        out = pd.DataFrame({"leaf": leaf, "confidence": np.round(np.clip(conf, 0, 1), 4)})
        for lvl in range(4):
            out[f"l{lvl + 1}"] = [leaves[i][lvl] if len(leaves[i]) > lvl else "" for i in leaf]
        return out
