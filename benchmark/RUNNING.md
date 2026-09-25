# Running the benchmark

How to reproduce the numbers in [README.md](README.md), point the benchmark at your
own models and datasets, and find your way around the output.

Everything is driven by [`config.yml`](config.yml) and every number and figure in
the output is generated. Nothing is typed by hand.

## Run it

```bash
cd benchmark
make setup       # uv sync --extra hf --group bench
make smoke       # a minute on a stand-in model: checks the pipeline, downloads no weights
make all         # the full sweep, then the cross-model analysis
```

Without uv, the same thing by hand:

```bash
pip install -e ".[hf]" datasets matplotlib pyyaml
python benchmark/run.py
python benchmark/analyze.py
```

`make` on its own lists every target. From the repository root, prefix them with
`-C benchmark`.

One dataset, or one model, while iterating:

```bash
make dataset-ag_news
make model-qwen3.5-2b
make all DATASETS="ag_news trec" MODELS=qwen3.5-2b
```

The keys are the ones in `config.yml` — the Makefile never names a model or a
dataset, so adding one needs no change here. `make list` prints them.

Rebuild the tables and plots from an existing run, with no model involved:

```bash
make report      # per-dataset reports, then results/analysis/
make analysis    # only results/analysis/
```

Everything takes `CONFIG=`, so a second config is a first-class citizen:
`make all CONFIG=ablation.yml`. Without `make`:

```bash
uv sync --extra hf --group bench
uv run --group bench benchmark/run.py --models qwen3.5-2b --datasets ag_news
uv run --group bench benchmark/analyze.py
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

## Layout

```
benchmark/
├── README.md             results and findings
├── RUNNING.md            this file
├── config.yml            models, datasets, sizes, seeds: everything the run depends on
├── smoke.yml             stand-in model, for checking the pipeline
├── Makefile
├── run.py                inference (cached) + calibration sweep + per-dataset reports
├── report.py             rebuild the per-dataset reports from results.json
├── analyze.py            cross-model figures and tables from results.json
├── logitbench/           the library behind the three scripts
└── results/
    ├── results.json      every record, plus the config that produced it
    ├── report.md         settings, the raw-vs-calibrated summary, links
    ├── analysis/         cross-model figures (*.png) and tables.md
    └── <dataset>/
        ├── report.md     one table per model
        ├── plots/*.png   metric vs. calibration size, reliability diagrams
        └── scores/*.npz  cached model output
```

Inference happens once per (model, dataset) and lands in `scores/`, keyed by a
digest of everything that could change it. Re-runs, extra seeds and new
calibration sizes are then pure arithmetic — seconds, not GPU hours. Change the
prompt, the model or the sampling and the digest changes, so the scores are
recomputed. The model's `base_url` is part of that digest too, so pointing a
served model at a different server scores it again.

`results.json` stores each reliability histogram as per-bin sums, so the bins of
different seeds and datasets simply add up. `analyze.py` relies on that for the
pooled figures.

## Configuring

**A model** is one entry under `models:`, keyed by the name that appears in the
tables. `analyze.py` has a small `DISPLAY` map for prettier names in figures; a key
missing from it is shown as-is.

```yaml
  qwen3.5-2b:
    kind: huggingface          # local weights
    model: Qwen/Qwen3.5-2B
    load: { batch_size: 16 }   # -> HuggingFaceBackend.from_pretrained

  my-served-model:
    kind: api                  # anything speaking the OpenAI chat API
    model: Qwen/Qwen3-VL-8B-Instruct
    base_url: http://localhost:8000/v1
    api_key_env: MY_API_KEY    # read from the environment, never written down here
    api: { top_logprobs: 20, concurrency: 16 }
    extra_body: { chat_template_kwargs: { enable_thinking: false } }  # sent with every request
```

A served model is read through `top_logprobs`, so every option's first token has
to make the server's top-k. An option that misses it is floored at the lowest
returned log-probability and the run warns about it. With many options (dbpedia
has 14) and a cap of 20, that happens now and then.

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
- Every calibration set is drawn from the same pool (512 examples by default), so
  at n = 256 the sets of different seeds overlap by about half and the seed spread
  there understates the true variance.
- Bad configuration fails on the spot, with the offending value in the message.
  Nothing is silently skipped or substituted.
