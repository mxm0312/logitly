# logitly calibration benchmark.
#
#   make setup && make smoke && make all
#
# From the repository root, prefix everything with -C benchmark:
#
#   make -C benchmark dataset-ag_news
#
# CONFIG, MODELS and DATASETS override what a target runs; keys are the ones in
# config.yml, so nothing here needs updating when you add a model or a dataset.

UV       ?= uv
CONFIG   ?= config.yml
MODELS   ?=
DATASETS ?=

PY   = $(UV) run --group bench python
RUN  = $(UV) run --group bench run.py --config $(CONFIG)
ANALYZE = $(UV) run --group bench analyze.py --config $(CONFIG)
ONLY = $(if $(MODELS),--models $(MODELS)) $(if $(DATASETS),--datasets $(DATASETS))

.DEFAULT_GOAL := help
.PHONY: help setup list all smoke report analysis clean FORCE

help:  ## show this
	@grep -E '^[a-zA-Z_%-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  variables: CONFIG=$(CONFIG)  MODELS=\"...\"  DATASETS=\"...\""

setup:  ## install the environment (drop --extra hf if you only use served models)
	$(UV) sync --extra hf --group bench

list:  ## print the model and dataset keys this config defines
	@$(PY) -c "import yaml; c = yaml.safe_load(open('$(CONFIG)')); \
	print('models   :', '  '.join(c['models'])); \
	print('datasets :', '  '.join(c['datasets']))"

all:  ## the full sweep, or a slice of it via MODELS= / DATASETS=
	$(RUN) $(ONLY)
	$(ANALYZE)

dataset-%: FORCE  ## one dataset, every model, e.g. make dataset-ag_news
	$(RUN) --datasets $* $(if $(MODELS),--models $(MODELS))

model-%: FORCE  ## one model, every dataset, e.g. make model-qwen3.5-2b
	$(RUN) --models $* $(if $(DATASETS),--datasets $(DATASETS))

smoke:  ## dry run on a stand-in model: no weights, checks the whole pipeline
	$(UV) run --group bench run.py --config smoke.yml

report:  ## rebuild tables and plots from an existing run, no model loaded
	$(UV) run --group bench report.py --config $(CONFIG)
	$(ANALYZE)

analysis:  ## only the cross-model figures and tables in results/analysis/
	$(ANALYZE)

clean:  ## delete this config's output directory, cached scores included
	@dir=$$($(PY) -c "import pathlib, yaml; \
	print(pathlib.Path('$(CONFIG)').parent / yaml.safe_load(open('$(CONFIG)')).get('output_dir', 'results'))") \
		&& echo "rm -rf $$dir" && rm -rf "$$dir"

FORCE:
