PY := PYTHONPATH=src python
CFG ?= configs/base.yaml

.PHONY: test lint manifest metadata labels registry ingest train report all clean

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests || true
	$(PY) -m pytest tests/test_invariants.py -q -k "portability or rdd or spark or hardcoded"

metadata:                       ## R13: fetch the 2.19 MB non-.edf payload
	$(PY) -m seizure.cli --config $(CFG) metadata

manifest:                       ## R30: header-only pre-pass, ~4.2 MB
	$(PY) -m seizure.cli --config $(CFG) manifest

labels:                         ## R29: parse + cross-validate, assert 198
	$(PY) -m seizure.cli --config $(CFG) labels

registry:
	$(PY) -m seizure.cli --config $(CFG) registry

ingest:                         ## R11/R14: stream download -> extract -> delete raw
	$(PY) -W ignore -m seizure.cli --config $(CFG) ingest $(ARGS)

train:
	$(PY) -W ignore -m seizure.cli --config $(CFG) train --view aggregated --contrast

report:
	$(PY) -m seizure.cli --config $(CFG) report

all: metadata manifest labels ingest train report

clean:
	rm -rf build artifacts/report.* artifacts/detail_*.json
