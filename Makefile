PY := .venv/bin/python

.PHONY: all setup stimuli preprocess encoders pretrain align recon brain braintv report test clean

all: stimuli preprocess pretrain encoders align recon brain braintv report

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

stimuli:    ; $(PY) scripts/01_stimuli.py
preprocess: ; $(PY) scripts/02_preprocess.py
pretrain:   ; $(PY) scripts/04_pretrain_jepa.py --variant all
	          $(PY) scripts/04_pretrain_jepa.py --variant sub-02
encoders:   ; $(PY) scripts/03_encoders.py
align:      ; $(PY) scripts/05_align.py
recon:      ; $(PY) scripts/06_reconstruct.py --encoder spectral
brain:      ; $(PY) scripts/07_brain3d.py
braintv:    ; $(PY) scripts/08_braintv.py --sub sub-01 --run 01
report:     ; $(PY) scripts/09_report.py
test:       ; $(PY) -m pytest tests/ -q

serve: report
	cd site/public && python3 -m http.server 8000

clean:
	rm -rf outputs/cache/* outputs/figures/* outputs/video/*
