"""Score the main Pythia panel's BLiMP paradigms at checkpoints before step 128.

The main panel starts at step 128, and the seed panel shows that the size-dependent stretch shrinks
when steps 16 to 64 are added. This scores all 67 BLiMP paradigms at steps 16, 32, 64, 128, 256 and
512 for the sizes in SIZES. The main panel's 200-pair sample per paradigm wasn't released, so this
draws its own (numpy default_rng(0) per paradigm, 200 of 1,000 pairs); the overlap at steps 128 to
512 measures agreement with the released accuracies. A pair counts as correct when the full-sentence
log probability (sum over tokens 2..n, no BOS) of the good sentence exceeds the bad one's.
Writes early_<TAG>.csv (size, step, paradigm, acc, n) after every checkpoint.

Note: model.eval() below is torch inference mode.
"""
import os, json, time, gc
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = "/kaggle/tmp/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import get_dataset_config_names, load_dataset

TAG = "__TAG__"
SIZES = {"small": ["14m", "31m", "70m", "160m", "410m"], "1b": ["1b"]}[TAG]
STEPS = [16, 32, 64, 128, 256, 512]
N_PAIRS, BATCH = 200, 64
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
torch.set_num_threads(os.cpu_count() or 4)


def retry(fn, what, tries=4, wait=45):
    for t in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"[retry {t+1}/{tries}] {what}: {type(e).__name__}: {e}", flush=True)
            if t == tries - 1:
                raise
            time.sleep(wait)


def enc(tok, sents):
    e = tok(sents, return_tensors="pt", padding=True)
    return e["input_ids"], e["attention_mask"]


@torch.no_grad()
def logprobs(model, ids, am):
    out = []
    for i in range(0, len(ids), BATCH):
        x, m = ids[i:i + BATCH], am[i:i + BATCH]
        lp = torch.log_softmax(model(input_ids=x, attention_mask=m).logits.float()[:, :-1], -1)
        t = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        out.append((t * m[:, 1:].float()).sum(-1))
    return torch.cat(out)


tok = retry(lambda: AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"), "tokenizer")
tok.pad_token, tok.padding_side = tok.eos_token, "right"
paradigms = sorted(retry(lambda: get_dataset_config_names("nyu-mll/blimp"), "configs"))
print(len(paradigms), "paradigms", flush=True)
data, meta = {}, {}
for p in paradigms:
    ds = retry(lambda p=p: load_dataset("nyu-mll/blimp", p, split="train"), p)
    idx = np.sort(np.random.default_rng(0).choice(len(ds), N_PAIRS, replace=False))
    sub = ds.select(idx.tolist())
    data[p] = (enc(tok, [str(x) for x in sub["sentence_good"]]), enc(tok, [str(x) for x in sub["sentence_bad"]]))
    meta[p] = dict(n_total=len(ds), fingerprint=getattr(ds, "_fingerprint", None))
json.dump(dict(paradigms=meta, steps=STEPS, n_pairs=N_PAIRS, sample="default_rng(0).choice per paradigm, sorted"),
          open(os.path.join(OUT, f"early_{TAG}_meta.json"), "w"), indent=1)

path = os.path.join(OUT, f"early_{TAG}.csv")
rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
done = {(r["size"], r["step"]) for r in rows}
for size in SIZES:
    for step in STEPS:
        if (f"pythia-{size}", step) in done:
            continue
        t0 = time.time()
        model = retry(lambda: AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{size}", revision=f"step{step}",
                                                                   torch_dtype=torch.float32), f"{size}@{step}", wait=90)
        model.eval()
        for p, (g, b) in data.items():
            acc = float((logprobs(model, *g) > logprobs(model, *b)).float().mean())
            rows.append(dict(size=f"pythia-{size}", step=step, paradigm=p, acc=acc, n=N_PAIRS))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{size} step{step} done in {time.time()-t0:.0f}s mean acc "
              f"{np.mean([r['acc'] for r in rows if r['size']==f'pythia-{size}' and r['step']==step]):.3f}", flush=True)
        del model
        gc.collect()
        try:
            import shutil; shutil.rmtree("/kaggle/tmp/hf/hub", ignore_errors=True)
        except Exception:
            pass
print("finished", flush=True)
