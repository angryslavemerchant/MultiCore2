"""Four-stroke machine interpretability suite: what IS each machine?

    python scripts/interp_fourstroke.py --run-name d512-4stroke-...

Four probes over the trained population, all from one checkpoint:

  contexts   for every (block, machine), the top gate-opening moments:
             the (token, context) snippets where its write-back gate
             saturates. Decoded with the NeoX tokenizer when available,
             raw token ids otherwise.
  ablation   mute one machine k across ALL blocks and measure the eval
             loss delta, two ways: mute_write (its gate forced to 0 — it
             can think but not speak to the tokens) and mute_conf (its
             published k/v masked out of the conference — other machines
             cannot read it; it still writes). Speaker roles vs internal-
             service roles. Also reports the delta on the last quarter of
             positions (long-context reliance).
  lens       logit-lens: each machine's would-be write-back W_O(c_k)
             pushed through ln_f + lm_head — the tokens each machine
             "wants to say", aggregated over sampled positions.
  graph      who-reads-whom: the K x K conference attention averaged over
             tokens and heads, per block.

Pulls best.pt from the Drive bank if missing (bench-box friendly).
Results: runs/<run>/interp_fourstroke.json (+ push if rclone ready).
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank                                                    # noqa: E402
from core.fourstroke import MachineStrokes                     # noqa: E402

SNIP_BEFORE, SNIP_AFTER = 48, 8


class Ctl:
    """Shared control knobs for all patched blocks."""
    mute_write = None      # machine index whose gate is forced to 0
    mute_conf = None       # machine index masked out of conference keys
    collect = False        # gather stats (contexts/lens/graph passes)


def patched_forward(self, x, c_prev, ctl, rec):
    B, T, C = x.shape
    K, d = self.K, self.d
    s = self.backend(x, c_prev)
    sn = self.ln_iface(s)
    q, ks, v = self.w_qkv(sn).chunk(3, dim=-1)
    k = (self.anchor[None, :, None, :]
         + self.addr_mix[None, :, None, None] * ks)

    def heads(t):
        return (t.permute(0, 2, 1, 3)
                .reshape(B * T, K, self.H, self.hd).transpose(1, 2))

    scores = heads(q) @ heads(k).transpose(-2, -1) / math.sqrt(self.hd)
    if ctl.mute_conf is not None:
        scores[..., ctl.mute_conf] = float("-inf")
    attn = torch.softmax(scores, dim=-1)          # (B*T, H, K, K)
    y = attn @ heads(v)
    y = (y.transpose(1, 2).reshape(B, T, K, d).permute(0, 2, 1, 3))
    c = s + self.conf_out(y)
    g = torch.sigmoid(
        torch.einsum("btc,kc->btk", x, self.gate_x)
        + torch.einsum("bktd,kd->btk", c, self.gate_c)
        + self.gate_b)
    if ctl.mute_write is not None:
        g = g.clone()
        g[..., ctl.mute_write] = 0.0
    out = torch.einsum("btk,bktc->btc", g, self.w_o(c))
    if ctl.collect:
        with torch.no_grad():
            rec["g"] = g.detach().float()                     # (B,T,K)
            rec["graph"] = (rec.get("graph", 0)
                            + attn.detach().float().mean(dim=(0, 1)))
            rec["graph_n"] = rec.get("graph_n", 0) + 1
            if rec.get("lens_pos") is not None:
                rec["lens_wo"] = self.w_o(
                    c[:, :, rec["lens_pos"], :]).detach()     # (B,K,P,C)
    return out, c


def batches(corpus, rng, n_batches, bs, T, device):
    for _ in range(n_batches):
        starts = rng.integers(0, len(corpus) - T - 2, bs)
        tok = np.stack([np.asarray(corpus[s:s + T + 1], dtype=np.int64)
                        for s in starts])
        yield (starts,
               torch.from_numpy(tok[:, :-1]).to(device),
               torch.from_numpy(tok[:, 1:]).to(device))


def eval_loss(model, corpus, rng, args, device, amp):
    """(mean_loss, last_quarter_loss) over a FIXED eval set (rng seeded by
    caller identically for every ablation arm)."""
    tot, n, tot_lq, n_lq = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for _, x, y in batches(corpus, rng, args.eval_batches,
                               args.micro_bs, args.seq_len, device):
            with amp:
                logits, _ = model(x, y)
            lp = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.reshape(-1),
                reduction="none").view(y.shape)
            tot += lp.sum().item()
            n += lp.numel()
            lq = lp[:, -lp.shape[1] // 4:]
            tot_lq += lq.sum().item()
            n_lq += lq.numel()
    return tot / n, tot_lq / n_lq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--batches", type=int, default=16,
                    help="collection batches (contexts/lens/graph)")
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="batches per ablation arm")
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--top-snippets", type=int, default=8)
    ap.add_argument("--lens-positions", type=int, default=32,
                    help="sampled positions per sequence for the lens")
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        assert bank.try_pull(ckpt_path,
                             folder=f"multicore2-runs/{args.run_name}"), \
            f"no {ckpt_path} and pull failed"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from core.model import model_from_ckpt_config
    cfg, model = model_from_ckpt_config(ckpt["config"])
    sd = ckpt.get("model", ckpt.get("model_state", ckpt))
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    blocks = [m for m in model.modules() if isinstance(m, MachineStrokes)]
    assert blocks, "not a four-stroke checkpoint"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    K = blocks[0].K
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    ctl = Ctl()
    recs = [{} for _ in blocks]
    for i, b in enumerate(blocks):
        b.forward = MethodType(
            (lambda self, x, c, _r=recs[i]:
             patched_forward(self, x, c, ctl, _r)), b)

    from needle_probe import ensure_corpus
    corpus = ensure_corpus()
    T = min(args.seq_len, cfg.block_size)

    # ---- pass 1: collection (contexts, lens, graph) -----------------------
    print("[interp] pass 1: collection", flush=True)
    ctl.collect = True
    rng = np.random.default_rng(args.seed)
    lens_rng = np.random.default_rng(args.seed + 1)
    # top gate moments: per (block, machine) running best (val, corpus_pos)
    top = [[([], []) for _ in range(K)] for _ in blocks]
    lens_counts = [[Counter() for _ in range(K)] for _ in blocks]
    ln_f, lm_head = model.transformer.ln_f, model.lm_head
    with torch.no_grad():
        for starts, x, y in batches(corpus, rng, args.batches,
                                    args.micro_bs, T, device):
            pos = np.sort(lens_rng.choice(T, args.lens_positions,
                                          replace=False))
            for r in recs:
                r["lens_pos"] = torch.from_numpy(pos).to(device)
            with amp:
                model(x)          # logits head skipped (targets=None)
            for li, r in enumerate(recs):
                g = r.pop("g")                                # (B,T,K)
                for k in range(K):
                    v, idx = g[:, :, k].reshape(-1).topk(args.top_snippets)
                    bpos = idx.cpu().numpy()
                    cpos = starts[bpos // T] + bpos % T
                    vals, poss = top[li][k]
                    vals.extend(v.cpu().tolist())
                    poss.extend(cpos.tolist())
                    order = np.argsort(vals)[::-1][:args.top_snippets]
                    top[li][k] = ([vals[o] for o in order],
                                  [poss[o] for o in order])
                wo = r.pop("lens_wo")                         # (B,K,P,C)
                with amp:
                    lg = lm_head(ln_f(wo))
                ids = lg.argmax(-1).reshape(wo.shape[0], K, -1)
                for k in range(K):
                    lens_counts[li][k].update(
                        ids[:, k].reshape(-1).cpu().tolist())
    ctl.collect = False

    # decode snippets
    try:
        from transformers import AutoTokenizer
        tokzr = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    except Exception as e:
        print(f"[interp] no tokenizer ({e}); raw ids only", flush=True)
        tokzr = None

    def snippet(p):
        lo, hi = max(0, p - SNIP_BEFORE), min(len(corpus), p + SNIP_AFTER + 1)
        ids = np.asarray(corpus[lo:hi], dtype=np.int64).tolist()
        at = p - lo
        if tokzr is None:
            return {"ids": ids, "at": at}
        return {"text": (tokzr.decode(ids[:at]) + " ⟦" +
                         tokzr.decode([ids[at]]) + "⟧ " +
                         tokzr.decode(ids[at + 1:hi - lo]))}

    contexts = [[{"gate_vals": [round(v, 3) for v in top[li][k][0]],
                  "snips": [snippet(p) for p in top[li][k][1]]}
                 for k in range(K)] for li in range(len(blocks))]
    lens = [[[{"tok": (tokzr.decode([t]) if tokzr else t),
               "frac": round(c / max(1, sum(lens_counts[li][k].values())), 3)}
              for t, c in lens_counts[li][k].most_common(8)]
             for k in range(K)] for li in range(len(blocks))]
    graph = [(r["graph"] / r["graph_n"]).cpu().tolist() for r in recs]
    for r in recs:
        r["lens_pos"] = None

    # ---- pass 2: ablation matrix ------------------------------------------
    print("[interp] pass 2: ablation", flush=True)

    def fixed_rng():
        return np.random.default_rng(args.seed + 999)

    base, base_lq = eval_loss(model, corpus, fixed_rng(), args, device, amp)
    print(f"[interp] baseline loss {base:.4f} (last-q {base_lq:.4f})",
          flush=True)
    ablation = {"baseline": {"loss": base, "loss_lastq": base_lq},
                "mute_write": [], "mute_conf": []}
    for mode in ("mute_write", "mute_conf"):
        for k in range(K):
            setattr(ctl, mode, k)
            l, lq = eval_loss(model, corpus, fixed_rng(), args, device, amp)
            setattr(ctl, mode, None)
            ablation[mode].append({"machine": k,
                                   "dloss": round(l - base, 5),
                                   "dloss_lastq": round(lq - base_lq, 5)})
            print(f"[interp] {mode} m{k}: dloss {l - base:+.4f} "
                  f"lastq {lq - base_lq:+.4f}", flush=True)

    report = {"run": args.run_name, "n_machines": K,
              "n_blocks": len(blocks),
              "collect_tokens": args.batches * args.micro_bs * T,
              "ablation": ablation, "conference_graph": graph,
              "lens": lens, "contexts": contexts}
    out = os.path.join(run_dir, "interp_fourstroke.json")
    json.dump(report, open(out, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"[interp] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)",
          flush=True)
    if bank._rclone_ready():
        bank.push(out, folder=f"multicore2-runs/{args.run_name}")
    print("INTERP_FS_DONE", flush=True)


if __name__ == "__main__":
    main()
