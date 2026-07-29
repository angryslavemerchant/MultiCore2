"""Stream-probe lifetime generator.

One lifetime = one <=T token sequence of natural-prose docs about nonce
entities, with grade sites (payload spans) at CONTROLLED token
distances from the docs that establish them. Every grade site carries
condition tags, so one generator yields five probe slices:

  cue=verbatim   bind statement repeated verbatim up to the payload
                 (pure induction cue, comparable to the needle probe)
  cue=fuzzy      realize template - semantically disjoint phrasing
                 (retrieval must be semantic; templates leak-gated)
  update=True    the fact was overwritten mid-stream; grade BOTH the
                 live value and the stale one at the same site
  collide=True   a near-duplicate key (subject sharing a surname) with
                 a different value sits between bind and recur
  cue=h2_*       two-hop: recur names only the person; payload belongs
                 to their employer (requires stream_probe/h2 bank)
  ghost          entity never introduced - the floor/control at every
                 distance

Distances are planned, then MEASURED exactly (bind-doc end -> payload
start, in tokens) and reported per site; analysis buckets by actual
distance. Docs are tokenized independently and id-concatenated, so
spans are exact by construction (no cross-doc BPE merges).

Capacity knob: cfg.n_background bind-only facts occupy memory without
ever being graded.
"""
import json
import os
import random
from dataclasses import dataclass, field

from stream_probe import nonce

DISTANCES = (128, 256, 512, 1024, 2048, 3584)
RELATIONS = ("works_at", "lives_in", "makes", "based_in")
_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ProbeConfig:
    T: int = 4096
    distances: tuple = DISTANCES
    update_distances: tuple = (256, 1024)
    collide_distances: tuple = (256, 1024)
    h2_distances: tuple = (256, 1024)
    ghost_distances: tuple = (256, 1024, 2048)
    n_background: int = 8          # bind-only facts (capacity pressure)
    bank_part: str = "hold"        # realize split for graded sites
    slack: int = 24                # placement jitter tolerance, tokens


def load_bank(path=None, h2_path=None):
    bank = json.load(open(path or os.path.join(_DIR, "bank.json"),
                          encoding="utf-8"))
    h2 = h2_path or os.path.join(_DIR, "h2.json")
    if os.path.exists(h2):
        bank.update(json.load(open(h2, encoding="utf-8")))
    return bank


def _split(lst, part):
    cut = (3 * len(lst)) // 4
    return lst[:cut] if part == "train" else lst[cut:]


def _fill(tpl, subj, val):
    return (tpl.replace("{P}", subj).replace("{C}", subj)
               .replace("{V}", val))


def _terminal_stmts(stmts):
    """Statement templates whose payload is terminal (verbatim cues)."""
    out = [t for t in stmts
           if t.rstrip(".!?, ").endswith("{V}") or t.rstrip().endswith("{V}")]
    return out


class Tok:
    """Per-doc tokenizer wrapper: every doc encoded with a leading
    space (except position 0) so id-concatenation is exact."""

    def __init__(self, hf_tok):
        self.t = hf_tok

    def doc(self, text, first=False):
        return self.t(text if first else " " + text)["input_ids"]

    def with_span(self, pre, full, first=False):
        """(ids, span) for a doc whose payload is full[len(pre):]."""
        a = self.doc(pre.rstrip(), first)
        b = self.doc(full, first)
        if b[:len(a)] != a or len(b) == len(a):
            return None
        return b, (len(a), len(b))


class Lifetime:
    def __init__(self, rng: random.Random, bank, tok: Tok,
                 cfg: ProbeConfig = None):
        self.cfg = cfg or ProbeConfig()
        self.rng = rng
        self.bank = bank
        self.tok = tok
        self.ids = []
        self.sites = []      # dicts: rel, cue, kind, flags, planned_d,
        #                      actual_d, span, payload, (stale fields)
        self._build()

    # -- item planning ---------------------------------------------------
    def _plan(self):
        rng, cfg = self.rng, self.cfg
        items = []

        def rel_cycle():
            rels = list(RELATIONS)
            rng.shuffle(rels)
            i = 0
            while True:
                yield rels[i % len(rels)]
                i += 1

        rc = rel_cycle()
        for d in cfg.distances:
            items.append(dict(rel=next(rc), cue="fuzzy", d=d))
            items.append(dict(rel=next(rc), cue="verbatim", d=d))
        for d in cfg.update_distances:
            items.append(dict(rel=next(rc), cue="fuzzy", d=d, update=True))
        for d in cfg.collide_distances:
            items.append(dict(rel=next(rc), cue="fuzzy", d=d, collide=True))
        if "h2_city" in self.bank:
            for d in cfg.h2_distances:
                items.append(dict(rel="based_in", cue="h2_city", d=d))
                items.append(dict(rel="makes", cue="h2_prod", d=d))
        for d in cfg.ghost_distances:
            items.append(dict(rel=next(rc), cue="fuzzy", d=d, ghost=True))
        return items

    # -- doc construction ------------------------------------------------
    def _recur_doc(self, item):
        """(ids, span[, alt_ids, alt_span]) for the grade doc; the alt
        variant carries the stale value for update sites."""
        rng, bank = self.rng, self.bank
        rel, cue, subj = item["rel"], item["cue"], item["subj"]
        if cue == "verbatim":
            tpl = item["bind_tpl"]                 # exact repeat
        elif cue.startswith("h2"):
            tpl = rng.choice(_split(bank[cue], self.cfg.bank_part))
        else:
            tpl = rng.choice(_split(bank[rel]["realize"],
                                    self.cfg.bank_part))
        tpl = tpl[:tpl.index("{V}") + 3]   # payload strictly terminal:
        #   no trailing punctuation may enter (and dilute) the grade span
        pre = _fill(tpl, subj, "\x00").split("\x00")[0]
        main = self.tok.with_span(pre, _fill(tpl, subj, item["val"]))
        if main is None:
            return None
        if item.get("update"):
            alt = self.tok.with_span(pre, _fill(tpl, subj, item["stale"]))
            if alt is None:
                return None
            return main + alt
        return main

    def _filler_doc(self, subj=None):
        rng, bank = self.rng, self.bank
        if subj is None or rng.random() < 0.5:
            subj = (nonce.person(rng) if rng.random() < 0.6
                    else nonce.company(rng))
        return self.tok.doc(_fill(rng.choice(bank["filler"]), subj, "") + ".")

    def _bg_fact_doc(self):
        rng, bank = self.rng, self.bank
        rel = rng.choice(RELATIONS)
        subj = nonce.SUBJECT_KIND[rel](rng)
        val = nonce.PAYLOAD_KIND[rel](rng)
        tpl = rng.choice(_split(bank[rel]["stmt"], "train"))
        return self.tok.doc(_fill(tpl, subj, val) + ".")

    # -- assembly --------------------------------------------------------
    def _build(self):
        rng, cfg, bank = self.rng, self.cfg, self.bank
        items = self._plan()

        anchors = []   # (planned_pos, order_key, kind, item, payload_ids_fn)
        for item in items:
            rel = item["rel"]
            subj = nonce.SUBJECT_KIND[rel](rng)
            val = nonce.PAYLOAD_KIND[rel](rng)
            item.update(subj=subj, val=val)
            d = item["d"]
            recur_pos = rng.randint(d + 192, max(d + 320, cfg.T - 96))
            recur_pos = min(recur_pos, cfg.T - 96)
            item["recur_pos"] = recur_pos

            if item.get("ghost"):
                anchors.append((recur_pos, "recur", item))
                continue

            stmts = _split(bank[rel]["stmt"], "train")
            if item["cue"] == "verbatim":
                term = _terminal_stmts(stmts)
                tpl = rng.choice(term if term else stmts)
                item["bind_tpl"] = tpl
            else:
                tpl = rng.choice(stmts)
            bind_text = _fill(tpl, subj, val)

            if item["cue"].startswith("h2"):
                # bridge: P works_at C placed at recur - d (the varied,
                # tracked link); company attribute fact ~300 tokens
                # earlier than that.
                comp = nonce.company(rng)
                w_tpl = rng.choice(_split(bank["works_at"]["stmt"], "train"))
                a_tpl = rng.choice(_split(bank[rel]["stmt"], "train"))
                anchors.append((recur_pos - d, "bind",
                                dict(text=_fill(w_tpl, subj, comp),
                                     track=item)))
                anchors.append((max(0, recur_pos - d - 300), "bind",
                                dict(text=_fill(a_tpl, comp, val))))
            elif item.get("update"):
                stale = nonce.PAYLOAD_KIND[rel](rng)
                item["stale"] = stale
                u_tpl = rng.choice(_split(bank[rel]["update"], "train"))
                anchors.append((recur_pos - 2 * d, "bind",
                                dict(text=_fill(tpl, subj, stale))))
                anchors.append((recur_pos - d, "bind",
                                dict(text=_fill(u_tpl, subj, val),
                                     track=item)))
            else:
                anchors.append((recur_pos - d, "bind",
                                dict(text=bind_text, track=item)))
                if item.get("collide"):
                    twin = subj.split()[0][:-2] + "on " + subj.split()[-1]
                    c_tpl = rng.choice(stmts)
                    anchors.append((recur_pos - d // 2, "bind",
                                    dict(text=_fill(
                                        c_tpl, twin,
                                        nonce.PAYLOAD_KIND[rel](rng)))))
            anchors.append((recur_pos, "recur", item))

        anchors.sort(key=lambda a: (a[0], 0 if a[1] == "bind" else 1))

        ids = []
        for pos, kind, obj in anchors:
            while len(ids) < pos - cfg.slack:
                doc = (self._bg_fact_doc() if rng.random() < 0.3
                       else self._filler_doc())
                if len(ids) + len(doc) > cfg.T:
                    break
                ids.extend(doc)
            if kind == "bind":
                doc = self.tok.doc(obj["text"] + ".", first=not ids)
                if "track" in obj:
                    obj["track"]["bind_end"] = len(ids) + len(doc)
                ids.extend(doc)
            else:
                item = obj
                r = self._recur_doc(item)
                if r is None:
                    continue
                doc, (s, e) = r[0], r[1]
                if len(ids) + len(doc) > cfg.T:
                    continue
                off = len(ids)
                ids.extend(doc)
                site = dict(
                    rel=item["rel"], cue=item["cue"],
                    kind="ghost" if item.get("ghost") else "real",
                    update=bool(item.get("update")),
                    collide=bool(item.get("collide")),
                    planned_d=item["d"],
                    actual_d=(off + s - item["bind_end"]
                              if "bind_end" in item else -1),
                    span=(off + s, off + e),
                    payload=item["val"])
                if item.get("update"):
                    # alt doc: same template/context, stale value.
                    # Eval grades it on ids[:off] + alt_ids (causal, so
                    # nothing after the doc matters).
                    alt_ids, (as_, ae) = r[2], r[3]
                    site.update(doc_off=off, alt_ids=alt_ids,
                                alt_span=(as_, ae),
                                stale_payload=item["stale"])
                self.sites.append(site)
        self.ids = ids


def build_lifetimes(n, seed, bank, hf_tok, cfg=None):
    tok = Tok(hf_tok)
    out = []
    for i in range(n):
        out.append(Lifetime(random.Random(seed + i), bank, tok, cfg))
    return out
