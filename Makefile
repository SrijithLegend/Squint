# Each target = one command. `make all` reproduces every table and figure from committed splits + configs.
# Long targets (train, eval) resume / skip finished work.
PY ?= python
SEEDS ?= 0 1 2

.PHONY: test data train quant eval bench stats report all clean-results
test:
	$(PY) -m pytest -q
data:                      # downloads the images listed in data/splits/*.json (needs data/coco/annotations/)
	$(PY) scripts/download_data.py
train:                     # resume-able; checkpoints every epoch in data/ckpt/
	for s in $(SEEDS); do $(PY) -m src.qrd.train $$s; done
quant:                     # ONNX export + PTQ INT8 (calibration from dev only)
	for s in $(SEEDS); do $(PY) -m src.qrd.quantize --seed $$s; done
eval:                      # test grid: 2 models x 17 conditions x 3 seeds (skips finished results)
	$(PY) -m src.qrd.evaluate --models fp32,ptq_int8 --seeds $(shell echo $(SEEDS) | tr ' ' ',') --split test
bench:
	$(PY) -m src.qrd.bench_latency
stats:
	$(PY) -m src.qrd.stats
report:
	$(PY) -m src.qrd.report
all: data train quant eval bench stats report
