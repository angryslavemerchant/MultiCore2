"""Stream world v2: full template bank, interleaved streams, fillers,
abstention (2026-07-22). Phase A of the end-to-end run Ibanis
authorized ("go for it... verify the whole pipeline and send the 124M
to an instance").

Upgrades over stream_text_icl v1:
- TEMPLATE BANK (templates_bank.json, agent-generated + validated):
  8 relations x ~44 statement paraphrases, 13 question types x ~16,
  124 fillers. Holdouts: last 10 statements/relation and last 4
  questions/type are EVAL-ONLY (paraphrase generalization measured on
  both sides of the interface).
- WORLD: 5 companies (founded/industry/based_in/makes) + 10 persons
  (works_as/lives_in/works_at) + 5 partner pairs = 55 facts/lifetime.
  Products are nonce names. Two-hop chains: founder->industry,
  founder->city, spouse->job, spouse->city, employer->industry.
- STREAM: single interleaved document stream — fact statements
  (2-3 paraphrases each) shuffled through, ~30% filler sentences,
  questions inserted with a MIN-GAP (>=25 docs after the asked fact's
  last statement; attention stays DOC-LOCAL so the book is the only
  channel regardless), final quiz block. Processed in chunks of 16
  docs (writes applied before each chunk's forward).
- ABSTENTION: ~10% of questions ask about entities that do not exist
  in this lifetime; correct answer is " unknown". Punishes
  over-retrieval; the read head must learn to signal absence.
- Metrics: per-hop exact match (multi-token), abstention accuracy,
  position-bucket curve, LM loss. Arms: live / live-theta / frozen /
  oracle / dense; eval also under held-out templates.

Wheels unchanged from v1 (fact-id filing, doc-mean-embedding
content); the learned writer belongs to the 124M graft (phase B).
Gate: dense ~ floor on everything incl. abstention-vs-known
discrimination; oracle/live >> dense; holdout ~= trained.
"""

import argparse
import json
import os
import random
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_stream_icl import Block, Book, ReadHead

ENC = tiktoken.get_encoding("gpt2")
L_DOC = 28
PAD, UNK = 0, 1
CHUNK = 16
MIN_GAP = 25

INDUSTRIES = ["shipping", "textiles", "robotics", "farming",
              "publishing", "insurance", "mining", "tourism"]
PROFESSIONS = ["baker", "lawyer", "pilot", "surgeon", "carpenter",
               "chemist", "fisherman", "architect"]
CITIES = ["Bristol", "Osaka", "Denver", "Turin", "Cairo", "Quebec",
          "Lagos", "Oslo", "Lisbon", "Perth", "Havana", "Kyoto"]
SYL_A = ["Vor", "Kre", "Tav", "Quen", "Mar", "Dor", "Bel", "Sar",
         "Fen", "Gal", "Hul", "Jor", "Lim", "Nav", "Pel", "Rud"]
SYL_B = ["lath", "ssin", "rix", "mar", "dan", "vek", "lor", "nis",
         "bram", "dell", "fort", "gen", "hart", "kell", "mond", "toll"]
CO_SUFFIX = ["Industries", "Group", "Logistics", "Works", "Holdings",
             "Systems"]
PROD_SUFFIX = ["Series", "Line", "Kit", "Blend", "Frame", "Core"]

N_C, N_P = 5, 10          # bisect knobs (main() may override)
STREAM_Q = 16
RELS = ["founded", "industry", "based_in", "makes",
        "works_as", "lives_in", "works_at", "partner"]
Q1 = ["q_founded", "q_industry", "q_based_in", "q_makes",
      "q_works_as", "q_lives_in", "q_works_at", "q_partner"]
Q2 = ["q_hop_industry", "q_hop_city", "q_hop_job", "q_hop_lives",
      "q_hop_works_industry"]

_BANK = None
_REMAP = None
_NVOCAB = None


def load_bank(path="templates_bank.json", cap=0):
    """cap>0: use only the first `cap` templates per category (bisect
    knob — v1-parity template diversity)."""
    global _BANK
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    _BANK = {"train": {}, "hold": {}}
    for k, items in raw.items():
        if k == "fillers":
            _BANK["train"][k] = items
            _BANK["hold"][k] = items
        elif k.startswith("q_"):
            tr = items[:-4]
            _BANK["train"][k] = tr[:cap] if cap else tr
            _BANK["hold"][k] = items[-4:]
        else:
            tr = items[:-10]
            _BANK["train"][k] = tr[:cap] if cap else tr
            _BANK["hold"][k] = items[-10:]
    return _BANK


def build_vocab(path="vocab_text_v2.json"):
    global _REMAP, _NVOCAB
    if os.path.exists(path):
        with open(path) as f:
            ids = json.load(f)
    else:
        rng = random.Random(7)
        texts = [" unknown", "unknown"]
        fill = dict(P="Aa Bb", P2="Cc Dd", C="Ee Ff Gg", I=INDUSTRIES[0],
                    X=CITIES[0], J=PROFESSIONS[0], PROD="Hh Ii")
        for part in ("train", "hold"):
            for k, items in _BANK[part].items():
                for t in items:
                    try:
                        texts.append(t.format(**fill))
                    except Exception:
                        texts.append(t)
        texts += [w for c in (INDUSTRIES, PROFESSIONS, CITIES)
                  for w in c]
        texts += [" " + w for c in (INDUSTRIES, PROFESSIONS, CITIES)
                  for w in c]
        for _ in range(40000):
            n = rng.choice(SYL_A) + rng.choice(SYL_B) + " " \
                + rng.choice(SYL_A) + rng.choice(SYL_B)
            texts += [n, " " + n]
        for _ in range(10000):
            n = rng.choice(SYL_A) + rng.choice(SYL_B)
            for sfx in CO_SUFFIX + PROD_SUFFIX:
                texts += [n + " " + sfx, " " + n + " " + sfx]
        ids = sorted({i for t in texts for i in ENC.encode(t)})
        with open(path, "w") as f:
            json.dump(ids, f)
    _REMAP = {g: i + 2 for i, g in enumerate(ids)}
    _NVOCAB = len(ids) + 2
    return _NVOCAB


def enc_c(text):
    return [_REMAP.get(i, UNK) for i in ENC.encode(text)]


UNKNOWN_IDS = None      # set after vocab build
_IDF = None             # per-token inverse doc frequency (v2.2)


def build_idf(n_sample=40):
    """v2.2: IDF weights for payload pooling. v2.0/2.1 gate failures
    traced to payload dilution — bank templates are long and varied,
    so a flat doc-mean is mostly function words; the answer token
    carries ~1/15 of the mass vs ~1/8 in v1's short templates. IDF
    pooling hands the mass to rare tokens (names, answers)."""
    global _IDF
    import math
    rng = random.Random(99)
    df = {}
    ndocs = 0
    for _ in range(n_sample):
        lt = Lifetime(rng)
        for (ids, *_rest) in lt.docs:
            ndocs += 1
            for t in set(ids):
                df[t] = df.get(t, 0) + 1
    idf = torch.ones(_NVOCAB)
    mx = math.log(ndocs + 1)
    idf *= mx          # unseen tokens (fresh names) get max weight
    for t, d in df.items():
        if t < _NVOCAB:
            idf[t] = math.log((ndocs + 1) / (d + 1)) + 0.1
    _IDF = idf
    return idf


def nonce_person(rng):
    return (rng.choice(SYL_A) + rng.choice(SYL_B) + " "
            + rng.choice(SYL_A) + rng.choice(SYL_B))


def nonce_co(rng):
    return rng.choice(SYL_A) + rng.choice(SYL_B) + " " \
        + rng.choice(CO_SUFFIX)


def nonce_prod(rng):
    return rng.choice(SYL_A) + rng.choice(SYL_B) + " " \
        + rng.choice(PROD_SUFFIX)


class Lifetime:
    """Fact graph + interleaved doc stream. Docs:
    (ids, kind, fid, aux1, aux2, ans_span)."""

    def __init__(self, rng, bank_part="train", stmts=2, filler_frac=0.3,
                 n_stream_q=None, n_quiz=26, abstain_frac=0.12):
        if n_stream_q is None:
            n_stream_q = STREAM_Q
        self.rng = rng
        B = _BANK[bank_part]
        seen = set()

        def fresh(gen):
            while True:
                x = gen(rng)
                if x not in seen:
                    seen.add(x)
                    return x
        self.persons = [fresh(nonce_person) for _ in range(N_P)]
        self.cos = [fresh(nonce_co) for _ in range(N_C)]
        self.prods = [fresh(nonce_prod) for _ in range(N_C)]
        self.founder = {c: self.persons[i]
                        for i, c in enumerate(self.cos)}
        self.industry = {c: rng.choice(INDUSTRIES) for c in self.cos}
        self.city = {c: rng.choice(CITIES) for c in self.cos}
        self.makes = {c: p for c, p in zip(self.cos, self.prods)}
        self.job = {p: rng.choice(PROFESSIONS) for p in self.persons}
        self.home = {p: rng.choice(CITIES) for p in self.persons}
        self.employer = {}
        for i, p in enumerate(self.persons):
            self.employer[p] = (self.cos[i] if i < N_C
                                else rng.choice(self.cos))
        order = list(self.persons)
        rng.shuffle(order)
        self.partner = {}
        for a, b in zip(order[0::2], order[1::2]):
            self.partner[a] = b
            self.partner[b] = a
        # enumerate facts -> fid; store (fid, texts)
        self.fid = {}
        facts = []

        def add(key, cat, **kw):
            self.fid[key] = len(facts)
            tpls = rng.sample(B[cat], min(stmts, len(B[cat])))
            facts.append([t.format(**kw) for t in tpls])
        for c in self.cos:
            add(("founded", c), "founded", P=self.founder[c], C=c)
            add(("industry", c), "industry", C=c, I=self.industry[c])
            add(("based_in", c), "based_in", C=c, X=self.city[c])
            add(("makes", c), "makes", C=c, PROD=self.makes[c])
        for p in self.persons:
            add(("works_as", p), "works_as", P=p, J=self.job[p])
            add(("lives_in", p), "lives_in", P=p, X=self.home[p])
            add(("works_at", p), "works_at", P=p, C=self.employer[p])
        done = set()
        for a in self.persons:
            if a not in done and a in self.partner:
                b = self.partner[a]
                done |= {a, b}
                add(("partner", frozenset((a, b))), "partner",
                    P=a, P2=b)
        self.n_facts = len(facts)
        # stream assembly: fact statements shuffled + fillers
        stmt_docs = []
        for f_i, texts in enumerate(facts):
            for t in texts:
                stmt_docs.append((enc_c(t), "fact", f_i, -1, -1, None,
                                  0))
        rng.shuffle(stmt_docs)
        n_fill = int(len(stmt_docs) * filler_frac / (1 - filler_frac))
        docs = list(stmt_docs)
        for _ in range(n_fill):
            t = rng.choice(B["fillers"])
            docs.insert(rng.randrange(len(docs) + 1),
                        (enc_c(t), "filler", -1, -1, -1, None, 0))
        # last-statement position per fact (for min-gap placement)
        last_pos = {}
        for pos, d in enumerate(docs):
            if d[1] == "fact":
                last_pos[d[2]] = pos
        # interleaved questions (position-curve instrument)
        qdocs = []
        for _ in range(n_stream_q):
            q = self._make_q(B, abstain_frac)
            if q is None:
                continue
            earliest = 0
            for f in q[7]:
                earliest = max(earliest, last_pos.get(f, 0) + MIN_GAP)
            if earliest < len(docs):
                pos = rng.randrange(earliest, len(docs) + 1)
                qdocs.append((pos, q[:7]))
        for pos, q in sorted(qdocs, key=lambda x: -x[0]):
            docs.insert(pos, q)
        # final quiz
        for _ in range(n_quiz):
            q = self._make_q(B, abstain_frac)
            if q is not None:
                docs.append(q[:7])
        self.docs = docs

    def _subj_pos(self, ids, subj):
        """Last token index of the subject mention (read-1 aux hook)."""
        for cand in (enc_c(" " + subj), enc_c(subj)):
            n = len(cand)
            if n == 0:
                continue
            for i in range(len(ids) - n, -1, -1):
                if ids[i:i + n] == cand:
                    return i + n - 1
        return max(0, len(ids) - 1)

    def _make_q(self, B, abstain_frac):
        rng = self.rng
        if rng.random() < abstain_frac:
            ghost = nonce_person(rng)
            qt = rng.choice(["q_works_as", "q_lives_in", "q_partner",
                             "q_works_at"])
            text = rng.choice(B[qt]).format(P=ghost)
            ids = enc_c(text)
            a = list(UNKNOWN_IDS)
            span = (len(ids), len(ids) + len(a))
            apos = self._subj_pos(ids, ghost)
            return (ids + a, "q_abstain", -1, -1, -1, span, apos, [])
        qt = rng.choice(Q1 + Q2)
        p = rng.choice(self.persons)
        c = rng.choice(self.cos)
        if qt == "q_founded":
            text, ans = rng.choice(B[qt]).format(C=c), self.founder[c]
            f1 = f2 = self.fid[("founded", c)]
        elif qt == "q_industry":
            text, ans = rng.choice(B[qt]).format(C=c), self.industry[c]
            f1 = f2 = self.fid[("industry", c)]
        elif qt == "q_based_in":
            text, ans = rng.choice(B[qt]).format(C=c), self.city[c]
            f1 = f2 = self.fid[("based_in", c)]
        elif qt == "q_makes":
            text, ans = rng.choice(B[qt]).format(C=c), self.makes[c]
            f1 = f2 = self.fid[("makes", c)]
        elif qt == "q_works_as":
            text, ans = rng.choice(B[qt]).format(P=p), self.job[p]
            f1 = f2 = self.fid[("works_as", p)]
        elif qt == "q_lives_in":
            text, ans = rng.choice(B[qt]).format(P=p), self.home[p]
            f1 = f2 = self.fid[("lives_in", p)]
        elif qt == "q_works_at":
            text, ans = rng.choice(B[qt]).format(P=p), self.employer[p]
            f1 = f2 = self.fid[("works_at", p)]
        elif qt == "q_partner":
            if p not in self.partner:
                return None
            text, ans = rng.choice(B[qt]).format(P=p), self.partner[p]
            f1 = f2 = self.fid[("partner", frozenset((p, ans)))]
        elif qt in ("q_hop_industry", "q_hop_city"):
            i = rng.randrange(N_C)
            p, c = self.persons[i], self.cos[i]
            text = rng.choice(B[qt]).format(P=p)
            ans = (self.industry[c] if qt == "q_hop_industry"
                   else self.city[c])
            f1 = self.fid[("founded", c)]
            f2 = self.fid[(("industry" if qt == "q_hop_industry"
                            else "based_in"), c)]
        elif qt in ("q_hop_job", "q_hop_lives"):
            if p not in self.partner:
                return None
            sp = self.partner[p]
            text = rng.choice(B[qt]).format(P=p)
            ans = self.job[sp] if qt == "q_hop_job" else self.home[sp]
            f1 = self.fid[("partner", frozenset((p, sp)))]
            f2 = self.fid[(("works_as" if qt == "q_hop_job"
                            else "lives_in"), sp)]
        else:                                   # q_hop_works_industry
            c = self.employer[p]
            text = rng.choice(B[qt]).format(P=p)
            ans = self.industry[c]
            f1 = self.fid[("works_at", p)]
            f2 = self.fid[("industry", c)]
        ids = enc_c(text)
        a = enc_c(" " + ans)
        span = (len(ids), len(ids) + len(a))
        hop = 2 if qt in Q2 else 1
        subj = c if qt in ("q_founded", "q_industry", "q_based_in",
                           "q_makes") else p
        apos = self._subj_pos(ids, subj)
        return (ids + a, f"q_h{hop}", -1, f1, f2, span, apos,
                [f1, f2])


def build_batch(B, device, rng, bank_part="train", stmts=2,
                filler_frac=0.3, abstain_frac=0.12, n_stream_q=None):
    lts = [Lifetime(rng, bank_part=bank_part, stmts=stmts,
                    filler_frac=filler_frac,
                    abstain_frac=abstain_frac,
                    n_stream_q=n_stream_q) for _ in range(B)]
    D = max(len(lt.docs) for lt in lts)
    toks = torch.full((B, D, L_DOC), PAD, dtype=torch.long)
    fids = torch.full((B, D), -1, dtype=torch.long)
    aux1 = torch.full((B, D), -1, dtype=torch.long)
    aux2 = torch.full((B, D), -1, dtype=torch.long)
    ans_mask = torch.zeros(B, D, L_DOC, dtype=torch.bool)
    q_pos = torch.zeros(B, D, dtype=torch.long)
    s_pos = torch.zeros(B, D, dtype=torch.long)
    kinds = [[None] * D for _ in range(B)]
    for b, lt in enumerate(lts):
        for d, (ids, kind, fid, a1, a2, span, apos) in \
                enumerate(lt.docs):
            ids = ids[:L_DOC]
            toks[b, d, :len(ids)] = torch.tensor(ids)
            fids[b, d] = fid
            aux1[b, d] = a1
            aux2[b, d] = a2
            kinds[b][d] = kind
            if span is not None:
                s, e = span
                e = min(e, L_DOC)
                ans_mask[b, d, s:e] = True
                q_pos[b, d] = s - 1
                s_pos[b, d] = min(apos, L_DOC - 1)
    return (toks.to(device), fids.to(device), aux1.to(device),
            aux2.to(device), ans_mask.to(device), q_pos.to(device),
            s_pos.to(device), kinds, lts[0].n_facts)


class TextStreamLM(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, use_book=True,
                 vocab=2000):
        super().__init__()
        self.d, self.use_book = d, use_book
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        pos = torch.zeros(L_DOC, d)
        t = torch.arange(L_DOC).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d))
        pos[:, 0::2] = torch.sin(t * div)
        pos[:, 1::2] = torch.cos(t * div)
        self.register_buffer("pos", pos * 0.02)
        m = torch.full((L_DOC, L_DOC), float("-inf"))
        self.register_buffer("mask", torch.triu(m, diagonal=1))
        self.blocks = nn.ModuleList(Block(d, heads)
                                    for _ in range(layers))
        self.read1 = ReadHead(d) if use_book else None
        self.read2 = ReadHead(d) if use_book else None
        self.ri = (2, layers - 1)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward_docs(self, toks, book=None):
        """toks (B,N,L) -> logits, w1, w2 (reads batched over B)."""
        Bb, N, L = toks.shape
        x = self.emb(toks.reshape(Bb * N, L)) + self.pos[:L]
        w1 = w2 = None
        for li, blk in enumerate(self.blocks):
            x = blk(x, self.mask[:L, :L])
            if self.use_book and book is not None:
                if li in self.ri:
                    xb = x.reshape(Bb, N * L, self.d)
                    if li == self.ri[0]:
                        xb, w1 = self.read1(xb, book)
                    else:
                        xb, w2 = self.read2(xb, book)
                    x = xb.reshape(Bb * N, L, self.d)
        logits = self.head(self.norm(x)).reshape(Bb, N, L, -1)
        shp = (Bb, N, L, -1)
        return logits, \
            (w1.reshape(shp) if w1 is not None else None), \
            (w2.reshape(shp) if w2 is not None else None)


def run_batch(model, batch, K, device, arm, aux_w, do_backward=False,
              loss_norm=40.0):
    """do_backward: call .backward() per CHUNK and free the graph —
    required at GPT-2 scale (the whole-lifetime graph OOMs 96 GB;
    every loss here is chunk-local because the book is non-
    differentiable, so per-chunk backward is exact, not an
    approximation). loss_norm: constant divisor replacing the
    end-of-lifetime question-count normalization."""
    (toks, fids, aux1, aux2, ans_mask, q_pos, s_pos, kinds,
     n_facts) = batch
    idf = _IDF.to(device) if _IDF is not None else None
    B, D, L = toks.shape
    book = None
    slots = torch.zeros(B, n_facts, dtype=torch.long, device=device)
    if model.use_book:
        book = Book(B, K, model.d, device)
        if arm == "frozen":
            book.counts[:] = 1.0
            book.pays = torch.randn_like(book.pays)
    loss = torch.zeros((), device=device)
    nq = 0
    stats = {k: [0.0, 0] for k in ("h1", "h2", "abstain")}
    curve = {}
    lm_sum, lm_n = 0.0, 0
    for c0 in range(0, D, CHUNK):
        chunk_loss = torch.zeros((), device=device)
        c1 = min(c0 + CHUNK, D)
        sub = toks[:, c0:c1]
        # writes for this chunk's fact docs (before its forward)
        if model.use_book and arm in ("live", "live-theta", "oracle"):
            with torch.no_grad():
                emb = model.emb(sub)
                npad = (sub != PAD).unsqueeze(-1).float()
                if idf is not None:
                    w = idf[sub].unsqueeze(-1) * npad
                else:
                    w = npad
                mean = (emb * w).sum(2) / w.sum(2).clamp(min=1e-6)
                for j in range(c1 - c0):
                    fcol = fids[:, c0 + j]
                    if not bool((fcol >= 0).any()):
                        continue
                    kv = F.normalize(mean[:, j], dim=-1)
                    pv = mean[:, j]
                    if arm == "live-theta":
                        i = book.write(kv, pv)
                    else:
                        i = fcol.clamp(min=0)
                        ar = torch.arange(B, device=device)
                        n = book.counts[ar, i] + 1
                        lr = (1.0 / n.clamp(max=book.cap)).unsqueeze(-1)
                        book.keys[ar, i] += lr * (kv - book.keys[ar, i])
                        book.pays[ar, i] += lr * (pv - book.pays[ar, i])
                        book.counts[ar, i] = n
                    ok = fcol >= 0
                    slots[ok, fcol[ok]] = i[ok] if i.dim() else i
        logits, w1, w2 = model.forward_docs(sub, book=book)
        tgt = sub[:, :, 1:]
        keep = tgt != PAD
        if bool(keep.any()):
            lm = F.cross_entropy(
                logits[:, :, :-1].reshape(-1, logits.shape[-1])
                [keep.reshape(-1)],
                tgt.reshape(-1)[keep.reshape(-1)])
            chunk_loss = chunk_loss + lm
            lm_sum += float(lm.detach())
            lm_n += 1
        amask = ans_mask[:, c0:c1, 1:] & keep
        if bool(amask.any()):
            ans_ce = F.cross_entropy(
                logits[:, :, :-1].reshape(-1, logits.shape[-1])
                [amask.reshape(-1)],
                tgt.reshape(-1)[amask.reshape(-1)])
            chunk_loss = chunk_loss + 2.0 * ans_ce
        # aux + metrics per question doc in chunk
        with torch.no_grad():
            pred = logits[:, :, :-1].argmax(-1)
            okt = (pred == tgt) | ~amask
            doc_ok = okt.all(-1) & amask.any(-1)
        for j in range(c1 - c0):
            d_i = c0 + j
            any_q = False
            for b in range(B):
                k = kinds[b][d_i]
                if k is None or not k.startswith("q_"):
                    continue
                any_q = True
                hit = float(doc_ok[b, j])
                key = ("abstain" if k == "q_abstain"
                       else ("h1" if k == "q_h1" else "h2"))
                stats[key][0] += hit
                stats[key][1] += 1
                bucket = min(d_i * 4 // max(D, 1), 3)
                cb = curve.setdefault((key, bucket), [0.0, 0])
                cb[0] += hit
                cb[1] += 1
            if any_q:
                nq += 1
        if model.use_book and arm != "frozen" and aux_w > 0 \
                and w1 is not None:
            acol1 = aux1[:, c0:c1]
            is_q = acol1 >= 0
            if bool(is_q.any()):
                bb, dd = torch.nonzero(is_q, as_tuple=True)
                pos = q_pos[:, c0:c1][bb, dd]
                spos = s_pos[:, c0:c1][bb, dd]
                s1 = slots[bb, acol1[bb, dd]]
                s2 = slots[bb, aux2[:, c0:c1][bb, dd]]
                # read1 hooked at the SUBJECT mention, read2 at the
                # pre-answer position (v1's working configuration)
                p1 = w1[bb, dd, spos, :].gather(
                    1, s1.unsqueeze(1)).squeeze(1)
                p2 = w2[bb, dd, pos, :].gather(
                    1, s2.unsqueeze(1)).squeeze(1)
                chunk_loss = chunk_loss - aux_w * (
                    torch.log(p1 + 1e-9)
                    + torch.log(p2 + 1e-9)).mean()
        if do_backward:
            if chunk_loss.requires_grad:
                (chunk_loss / loss_norm).backward()
            loss = loss + chunk_loss.detach()
        else:
            loss = loss + chunk_loss
    out = {k: v[0] / max(v[1], 1) for k, v in stats.items()}
    out["lm_loss"] = lm_sum / max(lm_n, 1)
    out["curve"] = {f"{k}_b{b}": v[0] / max(v[1], 1)
                    for (k, b), v in sorted(curve.items())}
    if book is not None:
        out["used"] = float((book.counts > 0).float().sum(1).mean())
    return loss / max(nq, 1), out


def train(model, steps, B, K, lr, device, tag, arm, aux_anneal, stmts,
          filler_frac, log_every=100, log_fn=None, abstain_frac=0.12,
          abstain_warmup=0.5, stream_q_warmup=0.0):
    """Curricula (both evidence-backed by the bisect campaign):
    - abstain_warmup: no abstention questions until this fraction of
      training — 'unknown' is a degenerate basin before the circuit
      exists (v2.0 failure).
    - stream_q_warmup: no MID-STREAM questions until this fraction —
      partial-book retrieval is the hardest regime and suppresses
      ignition 5x when present from step one (bisect D2 verdict:
      16/21 vs 85/97 on the single variable)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01)
    model.train()
    rng = random.Random(1234)
    t0 = time.time()
    for step in range(1, steps + 1):
        aux_w = max(0.0, 1.0 - step / max(steps * aux_anneal, 1)) \
            if (model.use_book and aux_anneal > 0) else 0.0
        af = abstain_frac if step > steps * abstain_warmup else 0.0
        sq = None if step > steps * stream_q_warmup else 0
        batch = build_batch(B, device, rng, stmts=stmts,
                            filler_frac=filler_frac, abstain_frac=af,
                            n_stream_q=sq)
        loss, st = run_batch(model, batch, K, device, arm, aux_w)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.3f}  "
                  f"lm {st['lm_loss']:.3f}  h1 {st['h1']:.3f}  "
                  f"h2 {st['h2']:.3f}  abst {st['abstain']:.3f}  "
                  f"aux {aux_w:.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            if log_fn:
                log_fn({f"{tag}/loss": loss.item(),
                        f"{tag}/h1": st["h1"], f"{tag}/h2": st["h2"],
                        f"{tag}/abstain": st["abstain"],
                        f"{tag}/step": step})


@torch.no_grad()
def eval_arm(model, B, K, device, arm, batches, stmts, filler_frac,
             bank_part="train", seed=999):
    model.eval()
    rng = random.Random(seed)
    agg, curve_agg = {}, {}
    for _ in range(batches):
        batch = build_batch(B, device, rng, bank_part=bank_part,
                            stmts=stmts, filler_frac=filler_frac)
        _, st = run_batch(model, batch, K, device, arm, 0.0)
        for k, v in st.items():
            if k == "curve":
                for kk, vv in v.items():
                    curve_agg.setdefault(kk, []).append(vv)
            else:
                agg.setdefault(k, []).append(v)
    out = {k: sum(v) / len(v) for k, v in agg.items()}
    out["curve"] = {k: sum(v) / len(v) for k, v in curve_agg.items()}
    return out


def main():
    global UNKNOWN_IDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--aux-anneal", type=float, default=0.4)
    ap.add_argument("--stmts", type=int, default=2)
    ap.add_argument("--filler-frac", type=float, default=0.3)
    ap.add_argument("--abstain-frac", type=float, default=0.12)
    ap.add_argument("--abstain-warmup", type=float, default=0.5)
    ap.add_argument("--n-c", type=int, default=5)
    ap.add_argument("--n-p", type=int, default=10)
    ap.add_argument("--bank-cap", type=int, default=0)
    ap.add_argument("--n-stream-q", type=int, default=16)
    ap.add_argument("--stream-q-warmup", type=float, default=0.0)
    ap.add_argument("--eval-batch", type=int, default=48)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-prefix", type=str, default="")
    ap.add_argument("--load-prefix", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str,
                    default="neocore-stream")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    global N_C, N_P, STREAM_Q
    N_C, N_P, STREAM_Q = args.n_c, args.n_p, args.n_stream_q
    load_bank(cap=args.bank_cap)
    nv = build_vocab()
    UNKNOWN_IDS = enc_c(" unknown")
    build_idf()
    print(f"device={device} vocab={nv} unknown={UNKNOWN_IDS} "
          f"idf built  world: {N_C}c/{N_P}p cap={args.bank_cap} "
          f"streamq={STREAM_Q}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"textv2-s{args.steps}-d{args.d}",
                         config=vars(args))
    log_fn = (lambda x: run.log(x)) if run else None

    kw = dict(d=args.d, layers=args.layers, vocab=nv)
    m_book = TextStreamLM(use_book=True, **kw).to(device)
    m_dense = TextStreamLM(use_book=False, **kw).to(device)
    if args.load_prefix:
        m_book.load_state_dict(torch.load(
            f"{args.load_prefix}_book.pt"))
        m_dense.load_state_dict(torch.load(
            f"{args.load_prefix}_dense.pt"))
        print("loaded", flush=True)
    else:
        train(m_book, args.steps, args.batch, args.k, args.lr, device,
              "book ", "live", args.aux_anneal, args.stmts,
              args.filler_frac, log_fn=log_fn,
              abstain_frac=args.abstain_frac,
              abstain_warmup=args.abstain_warmup,
              stream_q_warmup=args.stream_q_warmup)
        train(m_dense, args.steps, args.batch, args.k, args.lr,
              device, "dense", "live", 0, args.stmts,
              args.filler_frac, log_fn=log_fn,
              abstain_frac=args.abstain_frac,
              abstain_warmup=args.abstain_warmup,
              stream_q_warmup=args.stream_q_warmup)
    if args.save_prefix:
        torch.save(m_book.state_dict(),
                   f"{args.save_prefix}_book.pt")
        torch.save(m_dense.state_dict(),
                   f"{args.save_prefix}_dense.pt")

    arms = {}
    for arm, mdl in (("live", m_book), ("live-theta", m_book),
                     ("frozen", m_book), ("dense", m_dense)):
        arms[arm] = eval_arm(mdl, args.eval_batch, args.k, device,
                             arm, args.eval_batches, args.stmts,
                             args.filler_frac)
    arms["live-holdout"] = eval_arm(
        m_book, args.eval_batch, args.k, device, "live",
        args.eval_batches, args.stmts, args.filler_frac,
        bank_part="hold")
    arms["live-highfill"] = eval_arm(
        m_book, args.eval_batch, args.k, device, "live",
        args.eval_batches, args.stmts, 0.6)

    print("\nEXACT-match (h1/h2/abstain):")
    print("  metric  " + "  ".join(f"{n:>13s}" for n in arms))
    for m in ("h1", "h2", "abstain", "lm_loss"):
        row = "  ".join(f"{arms[n][m] * (1 if m == 'lm_loss' else 100):13.2f}"
                        for n in arms)
        print(f"  {m:>7s}  {row}")
    print("\nposition curve (live): "
          + json.dumps({k: round(v, 3)
                        for k, v in arms['live']['curve'].items()}))
    print(f"book used (live) {arms['live'].get('used', 0):.1f} "
          f"(facts=55); live-theta used "
          f"{arms['live-theta'].get('used', 0):.1f}", flush=True)

    if run:
        for a, st in arms.items():
            for k, v in st.items():
                if k != "curve":
                    run.summary[f"{a}_{k}"] = v
        with open("textv2_summary.json", "w") as f:
            json.dump({a: {k: v for k, v in st.items()}
                       for a, st in arms.items()}, f, indent=1)
        import wandb as wb
        art = wb.Artifact(f"textv2-{run.id}", type="results")
        art.add_file("textv2_summary.json")
        if args.save_prefix:
            art.add_file(f"{args.save_prefix}_book.pt")
            art.add_file(f"{args.save_prefix}_dense.pt")
        run.log_artifact(art).wait()
        run.finish()
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
