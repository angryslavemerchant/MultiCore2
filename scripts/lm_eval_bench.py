"""Run the standard small-LM zero-shot suite on a trained checkpoint via
EleutherAI's lm-evaluation-harness.

    python scripts/lm_eval_bench.py --run-name 124m-gpt2-causal-gelu
    python scripts/lm_eval_bench.py --run-name ... --tasks piqa,sciq

Wraps core.model.GPT + the NeoX tokenizer in the harness's loglikelihood
interface. Batches are RIGHT-padded: causality means trailing pads cannot
influence earlier positions (this holds for the gated layers too — a pad
admitted after position p can only shorten the lifetime of tokens as seen
by queries AFTER p, and we never read those), so no attention-mask support
is needed in the model.

Results append to runs/<run-name>/lm_eval.json.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig                          # noqa: E402

DEFAULT_TASKS = ("lambada_openai,hellaswag,piqa,winogrande,arc_easy,"
                 "arc_challenge,sciq,openbookqa,boolq")


def build_lm(ckpt_path, device, batch_size):
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    from transformers import AutoTokenizer

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    T = cfg.block_size
    print(f"loaded {ckpt_path}: step {ckpt['step']}, "
          f"val {ckpt.get('best_val'):.4f}, pattern "
          f"{cfg.attn_pattern or 'dense'}, block {T}")

    class OurLM(LM):
        def loglikelihood(self, requests):
            prepared = []
            for req in requests:
                ctx, cont = req.args
                ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"] \
                    if ctx else [tok.eos_token_id]
                cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
                ids = (ctx_ids + cont_ids)[-(T + 1):]
                n_cont = min(len(cont_ids), len(ids) - 1)
                prepared.append((ids, n_cont))

            out = [None] * len(requests)
            order = sorted(range(len(prepared)),
                           key=lambda i: -len(prepared[i][0]))
            with torch.no_grad():
                for s in range(0, len(order), batch_size):
                    chunk = order[s:s + batch_size]
                    L = max(len(prepared[i][0]) for i in chunk)
                    x = torch.zeros(len(chunk), L, dtype=torch.long)
                    for r, i in enumerate(chunk):
                        ids = prepared[i][0]
                        x[r, :len(ids)] = torch.tensor(ids)
                    x = x.to(device)
                    amp = (torch.autocast(device, dtype=torch.bfloat16)
                           if device == "cuda" else torch.no_grad())
                    with amp:
                        logits, _ = model(x[:, :-1], x[:, 1:])
                    logp = torch.log_softmax(logits.float(), dim=-1)
                    for r, i in enumerate(chunk):
                        ids, n_cont = prepared[i]
                        n = len(ids)
                        pos = range(n - 1 - n_cont, n - 1)
                        tgt = torch.tensor(ids[n - n_cont:], device=device)
                        lp = logp[r, list(pos), :]
                        s_lp = float(lp.gather(-1, tgt[:, None]).sum())
                        greedy = bool((lp.argmax(-1) == tgt).all())
                        out[i] = (s_lp, greedy)
            return out

        def loglikelihood_rolling(self, requests):
            res = []
            for req in requests:
                (text,) = req.args
                ids = tok(text, add_special_tokens=False)["input_ids"]
                ids = [tok.eos_token_id] + ids
                total = 0.0
                with torch.no_grad():
                    for s in range(0, len(ids) - 1, T):
                        seg = ids[s:s + T + 1]
                        x = torch.tensor(seg[:-1])[None].to(device)
                        y = torch.tensor(seg[1:])[None].to(device)
                        logits, _ = model(x, y)
                        lp = torch.log_softmax(logits.float(), -1)
                        total += float(
                            lp[0].gather(-1, y[0][:, None]).sum())
                res.append((total,))
            return res

        def generate_until(self, requests):
            raise NotImplementedError("MC suite only")

    return OurLM()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples per task (debugging)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import lm_eval
    lm = build_lm(os.path.join("runs", args.run_name, args.ckpt),
                  device, args.batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","),
                                  limit=args.limit)
    table = {t: {k: v for k, v in m.items() if isinstance(v, float)}
             for t, m in res["results"].items()}
    print(json.dumps(table, indent=2))
    out = os.path.join("runs", args.run_name, "lm_eval.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
