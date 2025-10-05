.PHONY: setup split train tune infer baseline-setfit baseline-tfidf plots clean

setup:
	uv venv
	source .venv/bin/activate && uv pip install -r requirements.txt

split:
	python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json --test_size 0.2 --folds 5

train:
	python -m src.train --config configs/lora_modernbert.yaml --fold 0

tune:
	python -m src.thresholding --preds runs/fold_0/val_preds.npy --labels runs/fold_0/val_labels.npy --out runs/fold_0/thresholds.json

infer:
	python -m src.inference --config configs/lora_modernbert.yaml --ckpts runs/fold_*/best.pt --ensemble mean --out runs/test_preds.csv

baseline-setfit:
	python -m src.setfit_baseline --config configs/setfit.yaml

baseline-tfidf:
	python -m src.tfidf_baseline --config configs/tfidf.yaml

plots:
	python -m src.plotting --preds runs/test_preds.csv --labels data/processed/test_labels.npy --out plots/

clean:
	rm -rf .venv runs plots data/processed/*
