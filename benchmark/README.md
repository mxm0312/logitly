# Calibration benchmark

Does fitting `softmax((z - b) / T)` on a handful of labelled examples actually
make a model's probabilities usable? This measures it: several models, several
classification sets, calibration sets from 8 to 256 examples, five seeds each.

Everything is driven by [`config.yml`](config.yml) and every number and figure in
the output is generated. Nothing is typed by hand.

## Run it

```bash
cd benchmark
make setup       # uv sync --extra hf --group bench
make smoke       # a minute on a stand-in model: checks the pipeline, downloads no weights
make all         # the full sweep
```

Without uv, the same thing by hand:

```bash
pip install -e ".[hf]" datasets matplotlib pyyaml
python benchmark/run.py
```

`make` on its own lists every target. From the repository root, prefix them with
`-C benchmark`.

One dataset, or one model, while iterating:

```bash
make dataset-sst2
make model-qwen2.5-7b-instruct
make all DATASETS="sst2 trec" MODELS=qwen2.5-1.5b-instruct
```

The keys are the ones in `config.yml` — the Makefile never names a model or a
dataset, so adding one needs no change here. `make list` prints them.

Rebuild the tables and plots from an existing run, with no model involved:

```bash
make report
```

Everything takes `CONFIG=`, so a second config is a first-class citizen:
`make all CONFIG=ablation.yml`. Without `make`:

```bash
uv sync --extra hf --group bench
uv run --group bench benchmark/run.py --models qwen2.5-1.5b-instruct --datasets sst2
```

## What it measures

For each model × dataset, on a fixed test set:

| | |
|---|---|
| accuracy | did calibration move the decisions |
| NLL | the quantity the fit minimises |
| ECE | 15 bins, top-label |
| Brier | proper score over the whole distribution |
| mean confidence | how sure the model says it is |
| predictions changed | share of test examples whose argmax the bias flipped |

`raw` is the uncalibrated softmax over the option logits. Every other row is a
calibration fitted on `n` examples the test set never sees, repeated over the
configured seeds and reported as mean ± std.

## Output

```
benchmark/results/
├── results.json          every record, plus the config that produced it
├── report.md             settings, the raw-vs-calibrated summary, links
└── <dataset>/
    ├── report.md         one table per model
    ├── plots/*.png       metric vs. calibration size, reliability diagrams
    └── scores/*.npz      cached model output
```

Inference happens once per (model, dataset) and lands in `scores/`, keyed by a
digest of everything that could change it. Re-runs, extra seeds and new
calibration sizes are then pure arithmetic — seconds, not GPU hours. Change the
prompt, the model or the sampling and the digest changes, so the scores are
recomputed.

## Configuring

**A model** is one entry under `models:`, keyed by the name that appears in the
tables:

```yaml
  qwen2.5-1.5b-base:
    kind: huggingface          # local weights
    model: Qwen/Qwen2.5-1.5B
    load: { batch_size: 16 }   # -> HuggingFaceBackend.from_pretrained
    prompt: { chat: false }    # -> PromptConfig; a base model has no chat template

  vllm:
    kind: api                  # anything speaking the OpenAI chat API
    model: Qwen/Qwen2.5-7B-Instruct
    base_url: http://localhost:8000/v1
    api_key_env: VLLM_API_KEY  # read from the environment, never written down here
```

**A dataset** is one entry under `datasets:`. `options` is indexed by the
dataset's own label id, so its order is not cosmetic — a mismatch in length is an
error, and so is a label id outside it.

```yaml
  ag_news:
    path: fancyzhx/ag_news     # -> datasets.load_dataset
    text_field: text
    label_field: label
    test_split: test
    calib_split: train
    options: [world, sports, business, science and technology]
    instructions: Which section of the newspaper does this story belong to?
```

The test set and the calibration pool are drawn once, with `data_seed`, and never
overlap, so every model answers exactly the same examples.

## Notes

- `.python-version` pins 3.12: the ML stack does not have wheels for every
  interpreter yet, and a benchmark that resolves differently per machine is not
  a benchmark.
- The test set keeps each dataset's natural class mix. Calibration sets follow it
  too unless `balanced_calibration: true`, which spreads them evenly over the
  classes instead.
- Bad configuration fails on the spot, with the offending value in the message.
  Nothing is silently skipped or substituted.
