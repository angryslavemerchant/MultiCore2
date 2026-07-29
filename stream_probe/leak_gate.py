"""Leak gate + ignition check for the stream-probe template bank.

Two pre-registered checks, run before the bank is frozen (re-run on
every bank edit):

A. LEAK GATE - each realize template, filled with nonce entities and
   presented ALONE (no stream context), must leave the payload tokens
   unpredictable: high NLL, uniformly across templates. A low-NLL
   template is hinting at its answer and must be rewritten.

B. IGNITION CHECK - the delta instrument must fire zero-shot at short
   range: [statement + fillers + realize] must give lower payload NLL
   than the same assembly with a statement about a DIFFERENT entity.
   If no arm can retrieve at ~100-token distance, the task is too hard
   for zero-shot 124M consumers and the design is void.

CPU-friendly: a few hundred short forward passes through one 124M
checkpoint (default: the phase-1 dense run).
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig            # noqa: E402
from stream_probe import nonce                   # noqa: E402

RELATIONS = ("works_at", "lives_in", "makes", "based_in")


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def fill(template, subj, val):
    return (template.replace("{P}", subj).replace("{C}", subj)
                    .replace("{V}", val))


def payload_spans(tok, texts_pre, texts_full):
    """Token ids + [start, end) payload span per item. Items where the
    BPE merges across the payload boundary are dropped (returned None)."""
    out = []
    for pre, full in zip(texts_pre, texts_full):
        ids_pre = tok(pre)["input_ids"]
        ids_full = tok(full)["input_ids"]
        if ids_full[:len(ids_pre)] != ids_pre or len(ids_full) == len(ids_pre):
            out.append(None)
            continue
        out.append((ids_full, len(ids_pre), len(ids_full)))
    return out


@torch.no_grad()
def batch_nll(model, items, device, batch_size=16):
    """Mean per-token NLL over each item's payload span."""
    nlls = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        L = max(len(ids) for ids, _, _ in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long, device=device)
        for j, (ids, _, _) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        logits, _ = model(x, x)
        logp = F.log_softmax(logits.float(), dim=-1)
        for j, (ids, s, e) in enumerate(chunk):
            tgt = torch.tensor(ids[s:e], device=device)
            lp = logp[j, s - 1:e - 1, :].gather(-1, tgt[:, None]).squeeze(-1)
            nlls.append(-lp.mean().item())
    return nlls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=os.path.join("stream_probe", "bank_draft.json"))
    ap.add_argument("--run-name", default="124m-gpt2-causal-gelu")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--samples", type=int, default=8,
                    help="nonce fills per template")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: this is a small local job")
    ap.add_argument("--gate-nll", type=float, default=6.0,
                    help="flag realize templates whose mean payload NLL "
                         "falls below this (leaky phrasing)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    bank = json.load(open(args.bank, encoding="utf-8"))
    tok = get_tokenizer()
    ckpt = torch.load(os.path.join("runs", args.run_name, args.ckpt),
                      map_location=args.device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    rng = random.Random(args.seed)
    report = {"run": args.run_name, "gate_nll": args.gate_nll,
              "leak": {}, "ignition": {}}

    # ---- A. leak gate: realize templates alone -------------------------
    print("== A. leak gate (realize templates, no context)")
    flagged = []
    for rel in RELATIONS:
        subj_gen, val_gen = nonce.SUBJECT_KIND[rel], nonce.PAYLOAD_KIND[rel]
        rows = []
        for ti, tpl in enumerate(bank[rel]["realize"]):
            pres, fulls = [], []
            for _ in range(args.samples):
                subj, val = subj_gen(rng), val_gen(rng)
                pres.append(fill(tpl, subj, "\x00").split("\x00")[0].rstrip())
                fulls.append(fill(tpl, subj, val))
            items = [it for it in payload_spans(tok, pres, fulls) if it]
            if not items:
                rows.append((ti, float("nan")))
                continue
            mean_nll = sum(batch_nll(model, items, args.device)) / len(items)
            rows.append((ti, mean_nll))
            if mean_nll < args.gate_nll:
                flagged.append((rel, ti, mean_nll, tpl))
        vals = [v for _, v in rows if v == v]
        report["leak"][rel] = {"per_template": rows,
                               "mean": sum(vals) / len(vals),
                               "min": min(vals)}
        print(f"  {rel:9s} mean {report['leak'][rel]['mean']:6.2f}  "
              f"min {report['leak'][rel]['min']:6.2f}  nats/token")
    for rel, ti, nll, tpl in flagged:
        print(f"  FLAG {rel}.realize[{ti}] NLL {nll:.2f}: {tpl[:70]}")

    # ---- B. ignition: stmt + fillers + realize vs mismatched stmt ------
    print("== B. ignition check (short-range zero-shot retrieval)")
    for rel in RELATIONS:
        subj_gen, val_gen = nonce.SUBJECT_KIND[rel], nonce.PAYLOAD_KIND[rel]
        n_stmt = len(bank[rel]["stmt"])
        deltas = []
        for _ in range(args.samples * 4):
            subj, val = subj_gen(rng), val_gen(rng)
            decoy_subj, decoy_val = subj_gen(rng), val_gen(rng)
            stmt = fill(rng.choice(bank[rel]["stmt"][: (3 * n_stmt) // 4]),
                        subj, val)
            decoy = fill(rng.choice(bank[rel]["stmt"][: (3 * n_stmt) // 4]),
                         decoy_subj, decoy_val)
            fillers = " ".join(fill(t, subj, "")
                               for t in rng.sample(bank["filler"], 2))
            tpl = rng.choice(bank[rel]["realize"])
            pre_tail = fill(tpl, subj, "\x00").split("\x00")[0].rstrip()
            body = fill(tpl, subj, val)
            pair = []
            for lead in (stmt, decoy):
                text_pre = f"{lead} {fillers} {pre_tail}"
                text_full = f"{lead} {fillers} {body}"
                it = payload_spans(tok, [text_pre], [text_full])[0]
                pair.append(it)
            if None in pair:
                continue
            with_nll, without_nll = batch_nll(model, pair, args.device)
            deltas.append(without_nll - with_nll)
        mean_d = sum(deltas) / len(deltas)
        se = (sum((d - mean_d) ** 2 for d in deltas)
              / (len(deltas) * (len(deltas) - 1))) ** 0.5
        report["ignition"][rel] = {"delta_mean": mean_d, "delta_se": se,
                                   "n": len(deltas)}
        print(f"  {rel:9s} delta {mean_d:+6.2f} +/- {se:.2f} nats/token "
              f"(n={len(deltas)})")

    out = os.path.join("stream_probe", "leak_gate_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
